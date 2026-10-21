"""AgentGate AWS layer (Phase 5): HTTP API -> router Lambda + trace Lambda, DynamoDB trace table
(TTL), S3 + CloudFront for the dashboard, a $5 budget alert. Thin on purpose: the engine runs on
the Mac and is reached through ENGINE_URL (Cloudflare Tunnel)."""
from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from constructs import Construct

LAMBDAS = Path(__file__).parent / "lambdas"
BUDGET_USD = 5


class AgentGateStack(Stack):
    def __init__(
        self, scope: Construct, id: str, *, engine_url: str, budget_email: str, **kwargs: object
    ) -> None:
        super().__init__(scope, id, **kwargs)

        table = dynamodb.Table(
            self, "Traces", table_name="agentgate-traces",
            partition_key=dynamodb.Attribute(name="trace_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="seq", type=dynamodb.AttributeType.NUMBER),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )

        def fn(name: str, handler: str, **env: str) -> lambda_.Function:
            return lambda_.Function(
                self, name, runtime=lambda_.Runtime.PYTHON_3_12, handler=handler,
                code=lambda_.Code.from_asset(str(LAMBDAS)), timeout=Duration.seconds(10),
                memory_size=128, environment=env,
            )

        router = fn("Router", "post_query.handler", ENGINE_URL=engine_url)
        trace = fn("Trace", "get_trace.handler", TRACE_TABLE=table.table_name)
        table.grant_read_data(trace)

        api = apigw.HttpApi(
            self, "Api", api_name="agentgate",
            cors_preflight=apigw.CorsPreflightOptions(
                allow_origins=["*"], allow_methods=[apigw.CorsHttpMethod.GET, apigw.CorsHttpMethod.POST],
                allow_headers=["content-type"],
            ),
        )
        api.add_routes(
            path="/query", methods=[apigw.HttpMethod.POST],
            integration=integrations.HttpLambdaIntegration("RouterInt", router),
        )
        api.add_routes(
            path="/trace/{trace_id}", methods=[apigw.HttpMethod.GET],
            integration=integrations.HttpLambdaIntegration("TraceInt", trace),
        )

        site = s3.Bucket(
            self, "Site", block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True,
        )
        cdn = cloudfront.Distribution(
            self, "Cdn", default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
        )

        budgets.CfnBudget(
            self, "Budget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="agentgate-5usd", budget_type="COST", time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=BUDGET_USD, unit="USD"),
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        comparison_operator="GREATER_THAN", notification_type=kind,
                        threshold=threshold, threshold_type="PERCENTAGE",
                    ),
                    subscribers=[budgets.CfnBudget.SubscriberProperty(
                        subscription_type="EMAIL", address=budget_email)],
                )
                for kind, threshold in (("ACTUAL", 80), ("FORECASTED", 100))
            ],
        )

        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "SiteUrl", value=f"https://{cdn.distribution_domain_name}")
        CfnOutput(self, "SiteBucket", value=site.bucket_name)
        CfnOutput(self, "TraceTable", value=table.table_name)
