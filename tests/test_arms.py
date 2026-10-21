import pytest

from engine.router.arms import ArmRegistry


def test_shipped_config_loads_and_resolves() -> None:
    reg = ArmRegistry.from_yaml()
    assert reg.get("qwen7b-rag").mode == "rag" and reg.get("qwen14b-agent").mode == "agent"


def test_unknown_and_duplicate_arms_fail_loudly() -> None:
    reg = ArmRegistry.from_yaml()
    with pytest.raises(KeyError, match="known"):
        reg.get("nope")
    with pytest.raises(ValueError, match="duplicate"):
        ArmRegistry(arms=reg.arms + reg.arms[:1])
