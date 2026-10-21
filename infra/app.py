"""`make synth` -> infra/cdk.out (template only). Deploy is a separate, approved step."""
import os
from pathlib import Path

from aws_cdk import App, Environment
from stack import AgentGateStack

app = App(outdir=os.environ.get("CDK_OUTDIR", str(Path(__file__).parent / "cdk.out")))
AgentGateStack(
    app, "AgentGate",
    engine_url=os.environ.get("ENGINE_URL", "http://localhost:8000"),
    budget_email=os.environ.get("BUDGET_EMAIL", "set-BUDGET_EMAIL@example.com"),
    env=Environment(region=os.environ.get("AWS_REGION", "ap-south-1")),
)
app.synth()
