from pathlib import Path
from types import SimpleNamespace
import json

import pytest

import scripts.verify_dppo_uav_rebuild as aggregate_verifier

from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.resume_probe import (
    verify_real_resume,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import (
    CheckpointHeader,
    SCHEMA_VERSION,
)


def _checkpoint(root: Path, update_index=1):
    root.mkdir()
    header = CheckpointHeader(
        schema_version=SCHEMA_VERSION,
        runtime_identity={},
        runtime_identity_hash="a" * 64,
        lora_identity_hash="b" * 64,
        update_index=update_index,
    )
    (root / "header.json").write_text(header.to_json(), encoding="utf-8")
    (root / "state.pt").write_bytes(b"state")
    (root / "SHA256.json").write_text("{}\n", encoding="utf-8")
    (root / "COMMITTED.json").write_text("{}\n", encoding="utf-8")
    return root


def test_resume_probe_is_read_only_and_reports_restored_counters(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("run:\n  updates: 1\n", encoding="utf-8")
    calls = []

    def runner(loaded_config, repo_root, resume_from):
        calls.append((loaded_config.run.updates, repo_root, resume_from))
        return SimpleNamespace(
            attempted_updates=1,
            committed_updates=1,
            rejected_updates=0,
            env_steps=128,
            completed_episodes=2,
            rollout_segments=3,
        )

    result = verify_real_resume(
        config,
        checkpoint,
        tmp_path,
        run_fn=runner,
    )

    assert result["status"] == "PASS"
    assert result["update_index"] == 1
    assert result["checkpoint_unchanged"] is True
    assert result["counters"]["env_steps"] == 128
    assert calls == [(1, tmp_path.resolve(), checkpoint.resolve())]


def test_resume_probe_refuses_a_config_that_would_continue_training(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("run:\n  updates: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="would continue training"):
        verify_real_resume(
            config,
            checkpoint,
            tmp_path,
            run_fn=lambda *args, **kwargs: None,
        )


def test_aggregate_verifier_runs_fast_contracts_and_optional_real_resume(
    monkeypatch, capsys
):
    commands = []
    monkeypatch.setattr(
        aggregate_verifier,
        "_run",
        lambda command, cwd: commands.append(tuple(str(item) for item in command)),
    )

    result = aggregate_verifier.main(
        [
            "--checkpoint",
            "e5.ckpt",
            "--device",
            "cuda:0",
            "--rl-config",
            "smoke.yaml",
            "--rl-checkpoint",
            "checkpoint_update_000001",
        ]
    )

    assert result == 0
    pytest_commands = [command for command in commands if "pytest" in command]
    assert len(pytest_commands) == 1
    assert "--ignore" in pytest_commands[0]
    resume_commands = [
        command
        for command in commands
        if any("verify_dppo_checkpoint_resume.py" in item for item in command)
    ]
    assert len(resume_commands) == 1
    assert "smoke.yaml" in resume_commands[0]
    assert "checkpoint_update_000001" in resume_commands[0]
    payload = json.loads(capsys.readouterr().out)
    assert payload["real_rl_checkpoint_resume"] == "PASS"
