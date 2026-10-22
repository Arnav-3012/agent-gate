"""`make synth` -> infra/cdk.out (template only). Deploy is a separate, approved step."""
import os
from pathlib import Path

from aws_cdk import App, Environment
from stack import AgentGateStack

budget_email = os.environ.get("BUDGET_EMAIL", "").strip()
if not budget_email:  # an empty subscriber address fails the whole deploy late (CloudFormation rollback)
    raise SystemExit("BUDGET_EMAIL is not set (see .env.example)")
app = App(outdir=os.environ.get("CDK_OUTDIR", str(Path(__file__).parent / "cdk.out")))
AgentGateStack(
    app, "AgentGate",
    engine_url=os.environ.get("ENGINE_URL", "http://localhost:8000"),
    engine_key=os.environ.get("ENGINE_KEY", ""),
    budget_email=budget_email,
    with_cdn=os.environ.get("WITH_CDN", "0") == "1",  # needs a CloudFront-verified account
    env=Environment(region=os.environ.get("AWS_REGION", "ap-south-1")),
)
app.synth()
