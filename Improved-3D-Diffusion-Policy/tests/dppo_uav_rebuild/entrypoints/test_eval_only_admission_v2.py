"""Production manifest ordering and eval-only exclusions are structural."""

from pathlib import Path


def test_production_roles_are_an_ordered_tuple_and_flatten_is_explicit():
    root = Path(__file__).resolve().parents[3]
    adapter = (root / "diffusion_policy_3d/rl/routea_rl_env_adapter_v1.py").read_text(encoding="utf-8")
    integration = (root / "diffusion_policy_3d/rl/dppo_uav/integration.py").read_text(encoding="utf-8")
    assert 'TRAINABLE_ROLES: Tuple[str, ...] = (' in adapter
    assert "for role in TRAINABLE_ROLES" in integration
    assert ".items()" not in integration[integration.index("def _flatten_entries"):integration.index("def manifest_train_entries")]
