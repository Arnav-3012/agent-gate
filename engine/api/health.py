"""Live-vs-replay decision from the /health payload (parallel-work C2); mirrored in dashboard/lib/health.ts."""
from typing import Literal

from pydantic import BaseModel

Mode = Literal["live", "replay", "unavailable"]


class Health(BaseModel):
    ok: bool
    llm: bool
    qdrant: bool


def decide_mode(health: Health | None, replay_available: bool) -> Mode:
    """Live only when the engine reports every dependency up. Anything else (engine unreachable =
    None, LLM down, Qdrant down) falls back to recorded traces; with none recorded there is nothing to show."""
    if health is not None and health.ok and health.llm and health.qdrant:
        return "live"
    return "replay" if replay_available else "unavailable"
