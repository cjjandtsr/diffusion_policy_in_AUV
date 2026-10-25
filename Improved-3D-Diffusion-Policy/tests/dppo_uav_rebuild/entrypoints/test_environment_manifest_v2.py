"""Security and reproducibility contract for the runtime environment manifest."""

from __future__ import annotations

import importlib
import json

import pytest


def _module():
    return importlib.import_module(
        "diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.environment_manifest"
    )


def test_environment_allowlist_never_copies_secrets_or_proxy_values():
    """Catches building the manifest by copying the full process environment."""

    environ = {
        "OMP_NUM_THREADS": "4",
        "NCCL_DEBUG": "INFO",
        "MASTER_ADDR": "10.0.0.9",
        "MASTER_PORT": "29517",
        "MY_SECRET": "xyz",
        "HF_TOKEN": "fake",
        "WANDB_API_KEY": "also-fake",
        "AWS_SECRET_ACCESS_KEY": "do-not-log",
        "HTTPS_PROXY": "https://user:pass@example.invalid",
    }
    safe = _module().collect_safe_environment(environ)
    assert safe == {
        "OMP_NUM_THREADS": "4",
        "NCCL_DEBUG": "INFO",
        "MASTER_ADDR_present": True,
        "MASTER_PORT": 29517,
    }
    serialized = json.dumps(safe, sort_keys=True)
    for forbidden in (
        "MY_SECRET",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "HTTPS_PROXY",
        "xyz",
        "fake",
        "do-not-log",
        "user:pass",
        "10.0.0.9",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize("value", ["", "0", "65536", "not-a-port"])
def test_invalid_master_port_fails_closed(value):
    """Catches accepting an ambiguous or invalid distributed rendezvous port."""

    with pytest.raises(ValueError, match="MASTER_PORT"):
        _module().collect_safe_environment({"MASTER_PORT": value})


def test_write_environment_manifest_records_runtime_without_secrets(
    tmp_path, monkeypatch
):
    """Catches missing runtime identity fields or a secret-bearing disk artifact."""

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("HF_TOKEN", "never-write-this")
    output = tmp_path / "environment_manifest.json"
    result = _module().write_environment_manifest(
        output, device="cpu", dtype="float32"
    )
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["schema_version"] == 1
    assert result["requested_device"] == "cpu"
    assert result["requested_dtype"] == "float32"
    assert result["python"]["version"]
    assert result["numpy"]["version"]
    assert result["torch"]["version"]
    assert result["environment"]["OMP_NUM_THREADS"] == "3"
    serialized = output.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in serialized
    assert "never-write-this" not in serialized
