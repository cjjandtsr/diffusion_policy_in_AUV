"""R2.1 SAMPLER-LIVE/SEED-RESUME/ROLE-PROP: EpisodeTaskSampler contract.

Guide_A_OfficialCore_R2_Sampler_Env_Contracts_v1_0.md Task R2.1 — the single
stateful sampler shared by the production train loop and the G3 gate audit:

- 10k sample proportions <= 2pp per role (RoleSampler gate);
- one world may serve multiple roles, all reachable through the sampler;
- save/load resumes the exact sequence (admission_id, seed) for 100 steps;
- consecutive batches never repeat global attempt/episode ids and seeds
  continue (no per-batch restart at 1);
- empty role bucket hard-fails at construction (no silent cross-role
  fallback);
- eval-only specs hard-fail at construction (no training leakage).
"""

import base64
from dataclasses import FrozenInstanceError
import pickle

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
    ROLE_NAMES,
    SampledEpisodeTask,
)

ROLE_NAMES_LIST = list(ROLE_NAMES)


def _spec(
    world: str,
    role: str,
    split: str = "rl_train",
    admission_id: str = None,
) -> EpisodeTaskSpec:
    return EpisodeTaskSpec(
        world_uid=world,
        obstacle_json=f"{world}.json",
        role=role,
        split=split,
        admission_id=admission_id or f"{world}__{role}",
        generator_version="routeA_rl_v1.3",
        manifest_entry_hash=f"hash_{world}_{role}",
    )


def _full_role_specs(per_role: int = 1) -> list:
    """One world per role (per_role copies of the same world for round-robin)."""
    specs = []
    for r in ROLE_NAMES_LIST:
        for k in range(per_role):
            specs.append(_spec(f"world_{r}_{k}", r))
    return specs


def _count_roles(sampler: EpisodeTaskSampler, n: int) -> dict:
    counts = {r: 0 for r in ROLE_NAMES_LIST}
    for _ in range(n):
        counts[sampler.next_attempt().task_spec.role] += 1
    return counts


# ---------- interface shape ----------

def test_sampled_episode_task_frozen():
    sample = SampledEpisodeTask(
        task_spec=_spec("w1", "anticipatory"),
        episode_seed=7,
        global_attempt_index=3,
        global_episode_index=2,
    )
    with pytest.raises(FrozenInstanceError):
        sample.episode_seed = 8  # type: ignore[misc]


# ---------- 10k proportion gate ----------

def test_10k_attempts_within_2pp():
    sampler = EpisodeTaskSampler(
        _full_role_specs(), DEFAULT_ROLE_TARGETS, base_seed=42
    )
    counts = _count_roles(sampler, 10000)
    assert sum(counts.values()) == 10000
    for role, target in DEFAULT_ROLE_TARGETS.items():
        actual = counts[role] / 10000
        assert abs(actual - target) <= 0.02, (
            f"{role}: actual {actual:.4f} vs target {target:.4f}"
        )


# ---------- same world, multiple roles ----------

def test_same_world_multiple_roles_all_sampled():
    """§4.8: role is not 1:1 with world — the sampler must reach every role
    entry of a shared world, never keying one spec by world_uid alone."""
    specs = [
        _spec("worldA", "anticipatory", admission_id="adm_A_antic"),
        _spec("worldA", "dense_recovery", admission_id="adm_A_dense"),
        _spec("worldB", "support_expansion", admission_id="adm_B_supp"),
        _spec("worldB", "yaw_lateral_recovery", admission_id="adm_B_yaw"),
    ]
    sampler = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0)
    seen = set()
    for _ in range(10000):
        seen.add(sampler.next_attempt().task_spec.admission_id)
    assert {"adm_A_antic", "adm_A_dense"} <= seen


def test_bucket_round_robin_covers_all_specs():
    """Two worlds in one role bucket: the round-robin cursor must reach both."""
    specs = [
        _spec("wA", "anticipatory", admission_id="A0"),
        _spec("wB", "anticipatory", admission_id="B0"),
        _spec("wC", "dense_recovery"),
        _spec("wD", "support_expansion"),
        _spec("wE", "yaw_lateral_recovery"),
    ]
    sampler = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=1)
    seen = set()
    for _ in range(10000):
        seen.add(sampler.next_attempt().task_spec.admission_id)
    assert {"A0", "B0"} <= seen


# ---------- seed determinism ----------

def test_episode_seed_determined_by_base_seed_and_attempt():
    """episode_seed = f(base_seed, attempt): deterministic, never restarts
    from 1 per batch, and different base seeds give different sequences."""
    s1 = EpisodeTaskSampler(_full_role_specs(), DEFAULT_ROLE_TARGETS, base_seed=10)
    s2 = EpisodeTaskSampler(_full_role_specs(), DEFAULT_ROLE_TARGETS, base_seed=10)
    s3 = EpisodeTaskSampler(_full_role_specs(), DEFAULT_ROLE_TARGETS, base_seed=11)
    for _ in range(50):
        a, b, c = s1.next_attempt(), s2.next_attempt(), s3.next_attempt()
        assert a.episode_seed == b.episode_seed
        assert a.episode_seed != c.episode_seed
        assert a.episode_seed == 10 + a.global_attempt_index


def test_two_batches_never_repeat_ids_or_seeds():
    """Batch 2 must continue the global counters — no per-batch restart."""
    sampler = EpisodeTaskSampler(_full_role_specs(), DEFAULT_ROLE_TARGETS, base_seed=5)
    batch1 = [sampler.next_attempt() for _ in range(20)]
    sampler.mark_completed()
    sampler.mark_completed()
    batch2 = [sampler.next_attempt() for _ in range(20)]

    att1 = [s.global_attempt_index for s in batch1]
    att2 = [s.global_attempt_index for s in batch2]
    assert len(set(att1 + att2)) == 40  # no attempt id repeated across batches
    assert att2[0] == att1[-1] + 1  # strict continuation
    assert att2 == list(range(20, 40))

    ep1 = [s.global_episode_index for s in batch1]
    ep2 = [s.global_episode_index for s in batch2]
    assert len(set(ep1 + ep2)) == 40  # episode ids also unique
    assert ep2[0] == ep1[-1] + 1

    seed1 = [s.episode_seed for s in batch1]
    seed2 = [s.episode_seed for s in batch2]
    assert seed2 == list(range(seed1[-1] + 1, seed1[-1] + 21))


# ---------- save / load resume ----------

def test_save_load_resumes_identical_sequence():
    """After load_state_dict, the next 100 (admission_id, seed) pairs are
    identical to the uninterrupted run (SEED-RESUME)."""
    specs = _full_role_specs(per_role=2)
    original = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=99)
    for _ in range(57):  # arbitrary mid-run checkpoint
        original.next_attempt()
    original.mark_completed()
    state = original.state_dict()

    resumed = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=99)
    resumed.load_state_dict(state)

    pairs_a = [
        (s.task_spec.admission_id, s.episode_seed)
        for s in (original.next_attempt() for _ in range(100))
    ]
    pairs_b = [
        (s.task_spec.admission_id, s.episode_seed)
        for s in (resumed.next_attempt() for _ in range(100))
    ]
    assert pairs_a == pairs_b

    # both samplers sit at the same counter point after the 100 pairs:
    # full state equality (57 initial attempts + 100 resumed, 1 completed),
    # and the restored sampler is NOT stuck at the pre-load counters.
    sa, sb = original.state_dict(), resumed.state_dict()
    assert sb["global_attempt_index"] == sa["global_attempt_index"] == 157
    assert sb["global_episode_index"] == sa["global_episode_index"] == 157
    assert sb["global_completed_index"] == sa["global_completed_index"] == 1
    assert sb["cursors"] == sa["cursors"]
    assert sb["global_attempt_index"] != state["global_attempt_index"]


def test_load_state_dict_rejects_wrong_spec_set():
    """Restoring state over a different task-spec set hard-fails (the resume
    hash gate is the R3 generalization of the same principle)."""
    specs_a = _full_role_specs()
    specs_b = [
        _spec("other_world_0", "anticipatory"),
        _spec("other_world_1", "dense_recovery"),
        _spec("other_world_2", "support_expansion"),
        _spec("other_world_3", "yaw_lateral_recovery"),
    ]
    original = EpisodeTaskSampler(specs_a, DEFAULT_ROLE_TARGETS, base_seed=3)
    for _ in range(10):
        original.next_attempt()
    state = original.state_dict()

    resumed = EpisodeTaskSampler(specs_b, DEFAULT_ROLE_TARGETS, base_seed=3)
    with pytest.raises(ValueError):
        resumed.load_state_dict(state)


def test_load_state_dict_rejects_wrong_base_seed():
    specs = _full_role_specs()
    original = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=3)
    for _ in range(10):
        original.next_attempt()
    state = original.state_dict()

    resumed = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=4)
    with pytest.raises(ValueError):
        resumed.load_state_dict(state)


# ---------- hard-fail boundaries ----------

def test_empty_role_bucket_hard_fails():
    """A role with no task specs is a config error — hard-fail at
    construction, never silently substitute another role's world."""
    restricted = [
        _spec("wA", "anticipatory"),
        _spec("wB", "dense_recovery"),
        _spec("wC", "support_expansion"),
    ]  # yaw_lateral_recovery missing
    with pytest.raises(ValueError):
        EpisodeTaskSampler(restricted, DEFAULT_ROLE_TARGETS, base_seed=0)


def test_eval_only_spec_hard_fails():
    """Evaluation-only specs must never enter a training sampler — the
    construction-time hard-fail makes leakage structurally impossible."""
    specs = _full_role_specs()
    # R7-8 ruling (2026-08-24): the literal eval-only split is rl_val
    # ("rl_eval" is the normative name for rl_val ∪ rl_test).
    specs.append(_spec("eval_world", "anticipatory", split="rl_val"))
    with pytest.raises(ValueError):
        EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0)


# ---------- RNG state is captured ----------

def test_state_dict_contains_rng_state():
    sampler = EpisodeTaskSampler(_full_role_specs(), DEFAULT_ROLE_TARGETS, base_seed=7)
    sampler.next_attempt()
    state = sampler.state_dict()
    assert state["schema"] == "episode_task_sampler/v1"
    assert state["base_seed"] == 7
    rng_state = state["role_sampler_rng"]
    # R2 P1-3: base64+pickle (weights_only-safe — the raw RandomState tuple
    # embeds a numpy array that torch.load(weights_only=True) rejects; the
    # sampler state now persists inside the checkpoint)
    assert isinstance(rng_state, str)
    assert set(state["cursors"]) == set(ROLE_NAMES_LIST)
    assert state["task_spec_set_hash"]
    # restoring from the captured RNG state reproduces the role stream
    rng = np.random.RandomState()
    rng.set_state(pickle.loads(base64.b64decode(rng_state.encode("ascii"))))
    first_role = sampler.next_attempt().task_spec.role
    next_role = ROLE_NAMES_LIST[int(rng.choice(len(ROLE_NAMES_LIST), p=[DEFAULT_ROLE_TARGETS[r] for r in ROLE_NAMES_LIST]))]
    assert first_role == next_role
