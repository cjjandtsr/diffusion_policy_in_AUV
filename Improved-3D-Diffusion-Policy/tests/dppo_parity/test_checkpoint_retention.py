"""S1-S4 (review R5.5 / §8.5): lightweight resume checkpoints, atomic
save, retention, disk preflight.

S1 lightweight: a resume checkpoint's policy_state contains NO base
model.* keys (the frozen base is re-materialized from the BC policy on
restore); a release checkpoint carries the FULL policy.
S2 atomic: CheckpointBuilder.save writes a tmp sibling and renames —
no partial file is ever observable.
S3 retention: rotating latest keeps only the newest N checkpoints.
S4 disk preflight: projected bytes + reserve > free hard-fails before
rollout.
"""

import os
import shutil
from pathlib import Path

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig, run_e2e_batch
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig
from scripts.train_dppo_uav_official import (
    _disk_preflight,
    _retain_checkpoints,
)

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv


def _specs():
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        EpisodeTaskSpec,
    )

    return [
        EpisodeTaskSpec(
            world_uid=f"mock_ckpt_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock_ckpt__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_ckpt_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]


def _runtime():
    return {
        "device": "cpu",
        "reward_version": "v16",
        "manifest_hash": "ckpt-gate",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "world_assets_hash": "ckpt-gate",
        "timestamp": "ckpt-gate",
    }


def _run(contract, tmp_path, kind):
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs()
    loop_config = LoopConfig(
        mode="one_update", n_episodes=1, max_ep_len=6, device="cpu",
        output_dir=str(tmp_path),
    )
    trainer_config = TrainerConfig(batch_size=8, device="cpu")
    res = run_e2e_batch(
        env, policy, critic, contract, loop_config, trainer_config, specs,
        torch.optim.Adam(list(policy.model.parameters()), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        {"lora_params": [], "n_adaptors": 0}, _runtime(),
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
        checkpoint_kind=kind,
    )
    return res["checkpoint_path"]


class _PrefixedModel(torch.nn.Module):
    """Wraps a policy model so its state_dict keys carry the model.
    prefix — the strip gate operates on policy state keys."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.model = inner

    def forward(self, **kwargs):
        return self.model(**kwargs)


def test_s1_resume_checkpoint_strips_base_keys(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    contract = RLSystemContract()

    def _run_prefixed(tmp_path, kind):
        policy = FakePolicy(contract)
        policy.model = _PrefixedModel(policy.model)
        env = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
        critic = ValueCritic(contract, PolicyObsEncoder(policy))
        specs = _specs()
        loop_config = LoopConfig(
            mode="one_update", n_episodes=1, max_ep_len=6, device="cpu",
            output_dir=str(tmp_path),
        )
        trainer_config = TrainerConfig(batch_size=8, device="cpu")
        res = run_e2e_batch(
            env, policy, critic, contract, loop_config, trainer_config,
            specs,
            torch.optim.Adam(list(policy.model.parameters()), lr=1e-3),
            torch.optim.Adam(critic.parameters(), lr=1e-3),
            {"lora_params": [], "n_adaptors": 0}, _runtime(),
            task_sampler=EpisodeTaskSampler(
                specs, DEFAULT_ROLE_TARGETS, base_seed=0
            ),
            checkpoint_kind=kind,
        )
        return res["checkpoint_path"]

    resume_path = _run_prefixed(tmp_path / "r", "resume")
    release_path = _run_prefixed(tmp_path / "l", "release")
    d_r = CheckpointBuilder.load(Path(resume_path))
    d_l = CheckpointBuilder.load(Path(release_path))
    assert d_r.checkpoint_kind == "resume"
    assert d_l.checkpoint_kind == "release"
    base_keys = [k for k in d_l.policy_state if k.startswith("model.")]
    assert base_keys, "release must carry the base model keys"
    assert not any(
        k.startswith("model.") for k in d_r.policy_state
    ), "resume checkpoint must not duplicate the frozen base"
    assert set(d_r.policy_state) < set(d_l.policy_state), (
        "the resume policy_state must be a strict subset of the release"
    )


def test_s2_atomic_save_no_partial_file(tmp_path):
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    contract = RLSystemContract()
    path = Path(_run(contract, tmp_path / "a", "resume"))
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp")), (
        "no tmp sibling may remain after a successful save"
    )


def test_s3_retention_keeps_newest(tmp_path):
    out = tmp_path / "r"
    out.mkdir(parents=True)
    for i in range(6):
        p = out / f"dppo_uav_envstep_{i:09d}_ppoupd_{i:03d}.pt"
        p.write_bytes(b"x" * (100 + i))
        p.touch()  # distinct mtimes
    removed = _retain_checkpoints(out, keep_latest=2)
    assert removed == 4
    assert len(list(out.glob("dppo_uav_*.pt"))) == 2


def test_s4_disk_preflight_hard_fails(tmp_path, monkeypatch):
    orig = shutil.disk_usage  # capture before patching (no recursion)

    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda p: orig(p)._replace(free=1 * 2**30),
    )
    with pytest.raises(RuntimeError):
        _disk_preflight(tmp_path, max_updates=1000)
    # plenty of space passes
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda p: orig(p)._replace(free=1000 * 2**30),
    )
    _disk_preflight(tmp_path, max_updates=10)


# ═══════════════════════════════════════════════════════════════════
# R5.8-D (2026-08-17): release protection via the metrics sidecar +
# torn-metrics tolerance.
# ═══════════════════════════════════════════════════════════════════


def test_release_checkpoint_protected_via_metrics_sidecar(tmp_path):
    """D2/N6: the metrics.jsonl sidecar is the cross-process truth for
    which checkpoints are releases — retention never rotates them, even
    when they are the oldest files."""
    import json as _json
    import time as _time

    from scripts.train_dppo_uav_official import _retain_checkpoints

    out = tmp_path / "run"
    out.mkdir()
    release = out / "dppo_uav_envstep_000000001_ppoupd_000001.pt"
    release.write_bytes(b"release")
    r1 = out / "dppo_uav_envstep_000000002_ppoupd_000002.pt"
    r1.write_bytes(b"resume1")
    r2 = out / "dppo_uav_envstep_000000003_ppoupd_000003.pt"
    r2.write_bytes(b"resume2")
    r3 = out / "dppo_uav_envstep_000000004_ppoupd_000004.pt"
    r3.write_bytes(b"resume3")
    # mtime ordering: release oldest, r3 newest
    now = _time.time()
    release.touch()
    _time.sleep(0.02)
    r1.touch()
    _time.sleep(0.02)
    r2.touch()
    _time.sleep(0.02)
    r3.touch()
    (out / "metrics.jsonl").write_text(
        _json.dumps({"checkpoint_kind": "release",
                     "checkpoint_path": str(release)}) + "\n",
        encoding="utf-8",
    )
    _retain_checkpoints(out, keep_latest=2)
    assert release.exists(), "release must never be rotated"
    survivors = [p.name for p in out.glob("dppo_uav_*.pt")]
    assert len(survivors) == 3, survivors  # release + newest 2 resume


def test_torn_metrics_trailing_line_tolerated(tmp_path):
    """H-4: a crash mid-write leaves a torn trailing line — the G7 gate
    reader reports it and uses the intact records, never a fatal
    JSONDecodeError. R6-C (§11.4.3): the intact record carries
    schema_version (a no-schema record would add a legacy problem)."""
    import json as _json

    from scripts.g7_gate_report import _load_metrics

    m = tmp_path / "metrics.jsonl"
    m.write_text(
        _json.dumps({"schema_version": "routeA_rl_env_v1_r6c",
                     "metrics": {"approx_kl": 0.1}}) + "\n" + '{"trun',
        encoding="utf-8",
    )
    records, problems = _load_metrics(m)
    assert len(records) == 1, records
    assert len(problems) == 1 and "truncated" in problems[0], problems
