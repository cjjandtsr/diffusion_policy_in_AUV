"""The frozen paired bank and the unique canonical artifact persistence.

- Freeze determinism: (spec, seed) -> the SAME sampled config on every
  freeze, so the eval-time replay reproduces the frozen
  episode_config_hash (the freeze check passes by construction and fails
  loudly on bank/manifest/world drift);
- Bank JSON round-trip: paired_bank_to_dict -> paired_bank_from_dict ->
  identical entries, and precise ValueError rejections (version,
  source_hash, required fields, role/split enums);
- The UNIQUE canonical artifact filename: env_steps/ppo_update/
  checkpoint_sha12 formatting;
- write_paired_eval_json: atomic tmp+os.replace write, refuse to
  overwrite different content, no-op on identical content.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    EvalArtifact,
    EpisodePairedRecord,
    build_eval_artifact,
    freeze_paired_bank,
    paired_bank_from_dict,
    paired_bank_to_dict,
    paired_eval_artifact_name,
    write_paired_eval_json,
)
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    CAPABILITY_METRIC_KEYS,
    GROUP_OF_ROLE,
    OPPORTUNITY_INVALID_REASONS,
    OfflineDensity,
)

from mock_uav_env import MockUAVEnv
from multi_slot_mock import MultiSlotMockUAVEnv


def _contract() -> RLSystemContract:
    return RLSystemContract()


def _specs(n: int = 2) -> list:
    """One spec per role (the first n)."""
    return [
        EpisodeTaskSpec(
            world_uid=f"mock_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ][:n]


def _freeze(n: int = 2, seeds=(1001, 1002), contract=None):
    contract = contract or _contract()
    return freeze_paired_bank(
        MockUAVEnv(contract), _specs(n), seeds
    )


def _step(i: int) -> dict:
    return {
        "step": i, "elapsed": 0.1 * i,
        "pos": [float(i), 0.0, 1.0], "z": 1.0, "yaw": 0.0,
        "goal_dist": 5.0, "clearance": 1.0, "hit_count": 4096,
        "action_norm": 1.0, "action_max_abs": 0.5,
        "done_reason": "none", "terminated": False, "truncated": False,
    }


def _caps_for(episodes: list) -> dict:
    """Valid capability schema for an arbitrary episode set — groups
    whose role is absent get n_episodes=0 (validator role alignment)."""
    caps = {}
    for group, role in GROUP_OF_ROLE.items():
        n = sum(1 for ep in episodes if ep.role == role)
        caps[group] = {
            key: (n if key == "n_episodes" else 0.0)
            for key in CAPABILITY_METRIC_KEYS[group]
        }
    return caps


def _not_informative_outcomes():
    """R6-EVAL-2 (§11.5.3): 最小合法五件套 —— artifact 必须携带三组
    五件套；零机会 NOT_INFORMATIVE 形状（gate 绝无捏造 value）。"""
    from diffusion_policy_3d.rl.dppo_uav.evaluator import CAPABILITY_METRICS

    return {
        group: {
            "opportunity_count": 0,
            "exposure_count": 0,
            "event_count": 0,
            "value": None,
            "validity_reason": OPPORTUNITY_INVALID_REASONS["no_opportunity"],
        }
        for group in CAPABILITY_METRICS
    }


def _artifact() -> EvalArtifact:
    trace = {"bc": [_step(i) for i in range(3)],
             "rl": [_step(i) for i in range(3)]}
    from diffusion_policy_3d.rl.dppo_uav.paired_rng import policy_noise_seed

    ep_hash = "e0" * 32
    ep = EpisodePairedRecord(
        episode_config={"world": "w1", "seed": 42,
                        "episode_config_hash": ep_hash},
        seed=42, world="w1", role="anticipatory",
        start=[0.0, 0.0, 0.0], goal=[1.0, 2.0, 3.0],
        disturbance={"events": []},
        bc={"success": True, "collision": False, "timeout": False, "done_reason": "reached", "out_of_bounds": False, "altitude_exceed": False,
            "terminated": True, "truncated": False, "terminal_category": "reached",
            "return": 95.0, "steps": 40},
        rl={"success": False, "collision": True, "timeout": False, "done_reason": "collision", "out_of_bounds": False, "altitude_exceed": False,
            "terminated": True, "truncated": False, "terminal_category": "collision",
            "return": -20.0, "steps": 12},
        trace=trace,
        policy_noise_seed=policy_noise_seed("bank-abc", ep_hash, 0),
    )
    return build_eval_artifact(
        episodes=[ep],
        checkpoint_hash="ckpt123", config_hash="cfg456",
        contract_hash="contract789", manifest_hash="manifestabc",
        world_assets_hash="world123",
        code_hash="codedef", reward_version="v16", exec_steps=4,
        observation_invalid_count=0, missing_weight_keys=[],
        unexpected_weight_keys=[], lora_checksum="lora123",
        timestamp="2026-08-10T10:00:00", device="cuda:0",
        capability_metrics=_caps_for([ep]),
        capability_outcomes=_not_informative_outcomes(),
        history_indices=[5, 6, 7], use_obstacle_projection=False,
        bank_id="bank-abc",
        offline_density=OfflineDensity(
            counts={(x, 0, 2): 5 for x in range(17)}
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# Step 2: the frozen paired bank
# ═══════════════════════════════════════════════════════════════════

def test_freeze_is_deterministic():
    """The freeze contract: (spec, seed) -> the SAME sampled config on
    every freeze — the eval-time replay reproduces the frozen hash."""
    specs = _specs()
    a = freeze_paired_bank(MockUAVEnv(_contract()), specs, (1001, 1002))
    b = freeze_paired_bank(MockUAVEnv(_contract()), specs, (1001, 1002))
    assert [e.episode_config_hash for e in a] == [
        e.episode_config_hash for e in b
    ]
    assert a[0].start == b[0].start and a[0].goal == b[0].goal
    assert a[0].start_yaw == b[0].start_yaw
    # a different seed -> a different episode (episode_seed is hashed),
    # so the frozen hash differs
    c = freeze_paired_bank(
        MockUAVEnv(_contract()), specs, (2001, 2002)
    )
    assert a[0].episode_config_hash != c[0].episode_config_hash


def test_freeze_binds_spec_identity_and_sampled_geometry():
    entries = _freeze()
    for entry, spec in zip(entries, _specs()):
        assert entry.to_spec() == spec
        assert entry.world_uid == spec.world_uid
        assert entry.role == spec.role
        assert entry.manifest_entry_hash == spec.manifest_entry_hash
        assert entry.start and entry.goal, "sampled geometry must be bound"
        assert len(entry.episode_config_hash) == 64
        assert entry.admission["reachability_pass"] is True


def test_freeze_probe_slot_closed():
    """The freeze probe closes its slot (abort_episode, DONE) — a leak
    would hard-fail the next reset on the ACTIVE guard."""
    env = MockUAVEnv(_contract())
    freeze_paired_bank(env, _specs(), (1001, 1002))
    assert env.slot_state(0) == "DONE", (
        "the probe slot must be closed after freeze"
    )
    env.reset(_specs(1), [1001])  # must not raise on the ACTIVE guard


def test_freeze_rejects_multislot_env():
    with pytest.raises(RuntimeError, match="n_envs"):
        freeze_paired_bank(
            MultiSlotMockUAVEnv(_contract(), n_envs=4), _specs(), (1001, 1002)
        )


def test_bank_json_roundtrip():
    """to_dict -> from_dict -> IDENTICAL entries (frozen dataclass
    equality — the JSON schema is lossless)."""
    entries = _freeze()
    d = paired_bank_to_dict(entries, source_hash="s" * 64,
                            creation_commit="c" * 40)
    assert d["bank_version"] == "3"
    assert d["bank_id"], "bank v3: every bank carries a binding bank_id"
    assert json.loads(json.dumps(d)) == d, "bank JSON must be serializable"
    assert paired_bank_from_dict(d) == entries


def test_bank_from_dict_rejections():
    """Precise ValueError messages on every structural violation — the
    CLI-level (exit 2) gate both entrypoints share."""
    import copy

    base = paired_bank_to_dict(_freeze(), "s" * 64, "c" * 40)
    stale = copy.deepcopy(base)
    stale["bank_version"] = "1"
    with pytest.raises(ValueError, match="bank_version"):
        paired_bank_from_dict(stale)
    no_src = copy.deepcopy(base)
    del no_src["source_hash"]
    with pytest.raises(ValueError, match="source_hash"):
        paired_bank_from_dict(no_src)
    no_commit = copy.deepcopy(base)
    del no_commit["creation_commit"]
    with pytest.raises(ValueError, match="creation_commit"):
        paired_bank_from_dict(no_commit)
    empty = copy.deepcopy(base)
    empty["entries"] = []
    with pytest.raises(ValueError, match="non-empty entries"):
        paired_bank_from_dict(empty)
    missing = copy.deepcopy(base)
    del missing["entries"][0]["start_yaw"]
    with pytest.raises(ValueError, match="start_yaw"):
        paired_bank_from_dict(missing)
    bad_role = copy.deepcopy(base)
    bad_role["entries"][0]["role"] = "not_a_role"
    with pytest.raises(ValueError, match="invalid role"):
        paired_bank_from_dict(bad_role)
    bad_split = copy.deepcopy(base)
    bad_split["entries"][0]["split"] = "not_a_split"
    with pytest.raises(ValueError, match="invalid split"):
        paired_bank_from_dict(bad_split)


# ═══════════════════════════════════════════════════════════════════
# Step 4: the unique canonical artifact filename + atomic write
# ═══════════════════════════════════════════════════════════════════

def test_artifact_filename_format():
    assert paired_eval_artifact_name(
        123, 4, "abcd1234ef56"
    ) == "paired_eval_envstep_000000123_ppoupd_000004_abcd1234ef56.json"
    assert paired_eval_artifact_name(0, 0, "x" * 12).startswith(
        "paired_eval_envstep_000000000_ppoupd_000000_"
    )


def test_write_paired_eval_json_atomic_and_no_overwrite(tmp_path):
    """Atomic tmp+os.replace: the file lands whole, no tmp litter;
    different content under the same name hard-fails (never overwrites);
    identical content is a no-op (resume-friendly)."""
    artifact = _artifact()
    path = str(tmp_path / "paired_eval_envstep_000000001_ppoupd_000001_x.json")
    # the artifact's own recorded hash is the caller-side expectation here
    # (the real call sites pass contract_hash_of(contract))
    write_paired_eval_json(artifact, path,
                           expected_contract_hash=artifact.contract_hash)
    assert Path(path).exists()
    assert not list(tmp_path.glob("*.tmp")), "no tmp litter after the write"
    assert json.loads(Path(path).read_text(encoding="utf-8")) == (
        artifact.to_dict()
    )
    # identical content -> no-op, no error
    write_paired_eval_json(artifact, path,
                           expected_contract_hash=artifact.contract_hash)
    # different content under the same canonical name -> hard-fail, and
    # the existing file stays untouched
    other = dataclasses.replace(artifact, device="cuda:1")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_paired_eval_json(other, path,
                               expected_contract_hash=artifact.contract_hash)
    assert json.loads(Path(path).read_text(encoding="utf-8")) == (
        artifact.to_dict()
    ), "the failed overwrite must leave the existing artifact intact"
