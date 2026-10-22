# AWS docs (AgentGate Phase 5)

One file per service. Read this first, then the service file you need. The step-by-step deploy runbook is [../todo-aws-now.md](../todo-aws-now.md). Code: `infra/stack.py`, `infra/app.py`, `infra/lambdas/`, `engine/tracing/dynamo.py`.

Account `250674680827`, region `ap-south-1` (Mumbai), CDK stack name `AgentGate`.

## Deployed state (2026-10-22)
| Piece | State |
|---|---|
| CDK bootstrap | done |
| Stack `AgentGate` | **deployed without CloudFront** (`WITH_CDN` unset): HTTP API, Router + Trace Lambdas, DynamoDB `agentgate-traces`, $5 budget |
| `ApiUrl` | `https://gnvxffloug.execute-api.ap-south-1.amazonaws.com` |
| Smoke test | engine 401 without key; `POST /query` -> trace in DynamoDB -> `GET /trace` `done:true` (RAG, 1111 tokens, $0.000112) |
| `GET /health` route | deployed, verified through AWS |
| `domain` forwarding (finance) | deployed, verified (10-K filings retrieved) |
| S3 + CloudFront | **blocked**: new account must be verified by AWS Support first ([s3-cloudfront.md](s3-cloudfront.md)) |
| Hardening in place | shared secret `ENGINE_KEY`, API throttle 10 rps / burst 20 |
| Story of how we got here | [deploy-log.md](deploy-log.md) (every error, cause, fix) |

## Services used and why

| Service | What it does here | Created by | File |
|---|---|---|---|
| IAM | `agentgate-deploy` user (CLI/CDK), Lambda execution roles, CDK bootstrap roles | by hand + CDK | [iam-sts.md](iam-sts.md) |
| STS | "who am I" check, CDK role assumption | implicit | [iam-sts.md](iam-sts.md) |
| Billing + Budgets | free-tier alerts, `agentgate-5usd` budget with email alerts | by hand + CDK | [billing-budgets.md](billing-budgets.md) |
| CloudFormation | deploys everything in the stack as one unit, makes teardown one command | CDK | [cdk-cloudformation.md](cdk-cloudformation.md) |
| CDK bootstrap (S3 staging bucket, ECR repo, SSM parameter, roles) | holds Lambda code zips for deploy | `cdk bootstrap` | [cdk-cloudformation.md](cdk-cloudformation.md) |
| Lambda | `Router` (POST /query) and `Trace` (GET /trace) | CDK | [lambda.md](lambda.md) |
| API Gateway (HTTP API) | public HTTPS front door, routes, CORS | CDK | [api-gateway.md](api-gateway.md) |
| DynamoDB | trace event store, polled by the dashboard | CDK | [dynamodb.md](dynamodb.md) |
| S3 | private bucket for the dashboard's static files | CDK | [s3-cloudfront.md](s3-cloudfront.md) |
| CloudFront | HTTPS CDN in front of S3, Origin Access Control | CDK | [s3-cloudfront.md](s3-cloudfront.md) |
| CloudWatch Logs | Lambda logs (automatic); EMF metrics not built yet | implicit | [cloudwatch.md](cloudwatch.md) |
| (not AWS) Cloudflare Tunnel | makes the Mac engine reachable by the Router Lambda | by hand | [tunnel.md](tunnel.md) |

Not used on purpose: VPC, EC2, ECS, RDS, Cognito, WAF, Route 53, Secrets Manager, LangChain-style managed AI services. The LLM, Qdrant and embeddings stay on the Mac.

## Request flow

```
Browser (CloudFront, S3 files)
  | POST /query {question, arm?, mode?}
  v
API Gateway HTTP API ---> Router Lambda: mints trace_id (uuid hex), POST ENGINE_URL/run
                                          |  (engine answers 202 immediately)
                                          v
                          Cloudflare quick tunnel ---> FastAPI engine on the Mac
                                                         runs arm (RAG-7B / 14B / agent)
                                                         DynamoSink.put_item per trace event
                                                         |
                                                         v
                                                      DynamoDB agentgate-traces
  | GET /trace/{id}?after=N  (every 500 ms)               ^
  v                                                       |
API Gateway ---> Trace Lambda: query by trace_id -------- +
                 returns {events (seq > after), done}
```

`GET /health` (Router Lambda) proxies the engine's `/health`; 502 when the tunnel or engine is down, which the dashboard reads as "not live" and flips to replay. Dashboard in AWS mode (`NEXT_PUBLIC_API_URL` set) calls `/query` instead of the engine's `/run`.

The id is minted by the Lambda, so the browser can start polling the instant POST returns. Until the engine writes its first item, "no items" and "unknown id" look the same, so the poller just keeps polling (`done:false`).

## Cost picture (demo traffic, a few hundred requests)

Mostly effectively $0, but be precise about what is and is not free tier:
- Lambda: 1M requests + 400k GB-s per month always free; we use a tiny fraction.
- HTTP API: about $1 per million requests after the free allowance (first 12 months of a classic account). Negligible.
- DynamoDB on-demand: pay per request (cents per million). The always-free allowance is 25 GB storage; on-demand request charges are not covered by the free provisioned capacity, so expect pennies, not literally zero.
- CloudFront: 1 TB/month and 10M requests free always. S3: pennies for a few MB.
- CloudWatch Logs: ingest is cheap but retention defaults to **never expire**; see [cloudwatch.md](cloudwatch.md).
- Budgets: the first budgets per account are free; verify on the billing page.
- Newer AWS accounts may be on the credit-based "free plan" instead of the classic 12-month free tier. Check Billing -> Free tier to see which one your account has; the numbers above are the classic ones.

Safety net: the $5/month budget emails at 80% actual and 100% forecast. A budget only notifies, it does not stop spending. Billing data lags by roughly 8-24 h.

## Teardown order
1. `npx aws-cdk destroy --all` (removes the stack: table, Lambdas, API, bucket contents + bucket, distribution, budget).
2. Optionally delete the `CDKToolkit` stack and the `cdk-hnb659fds-assets-*` bucket (empty it first).
3. Deactivate then delete the `agentgate-deploy` access key (and the unused `demo-1` key if nothing else needs it).
4. Check the CloudFormation, DynamoDB, S3, CloudFront, Lambda consoles in ap-south-1 (CloudFront is global) show nothing left.

## Known gaps
Collected per service; the cross-cutting ones: no end-user auth on the API (throttled and secret-protected downstream only), public tunnel (secret-protected), the engine writes DynamoDB with an admin key, no EMF metrics, CloudFront blocked pending account verification. Details in each file's "Gaps" section.
