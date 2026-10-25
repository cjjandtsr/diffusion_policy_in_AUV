"""R6-F reward scaling contract (F-07 → R6-TUNE-1, Guide §11.7.3-11.7.4).

The frozen vendor RunningRewardScaler (util/reward_scaling.py at vendor
commit cc7234a, PPG-based) is the behavior oracle — never the project's
own formulas. Coverage:

- math parity: the production port reproduces the vendor's scaled
  rewards, carried per-slot returns, and running mean/var/count exactly
  over a fixed multi-batch sequence (identical float64 arithmetic);
- first-flag derivation: episode_ids + per-slot last_episode_id memory
  produce the same flags the vendor receives from rollout done flags
  (cross-batch continuation at t=0 == vendor's leading done_venv);
- episode boundary semantics: a mid-batch episode reset drops the
  carried return for that slot while other slots continue;
- none branch: transform is identity, state is structured
  (method/version/step_count) — never an empty dict;
- cold resume: a loaded scaler continues the exact sequence
  (next scaled rewards AND next state == uninterrupted oracle);
- tamper: method/version/source/step_count/ret/mean/var/count/
  last_episode_id/extra-keys — any single mutation fails validation;
- gamma: scaler gamma must equal contract.gamma_env (single-source
  discount, mismatch is a hard construction error);
- checkpoint: RewardNormState carries the scaler state dict (roundtrip
  through the checkpoint form, validate -> load -> identical next batch).
"""

import copy

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.reward_norm import (
    REWARD_SCALE_STATE_VERSION,
    RewardNormState,
    RewardScaleConfig,
    RunningRewardScaler,
    load_scaler_state,
    validate_reward_scale_state,
)

VENDOR_ROOT = (
    "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
    "/third_party/dppo_official_cc7234a"
)
import sys

if VENDOR_ROOT not in sys.path:
    sys.path.insert(0, VENDOR_ROOT)

from util.reward_scaling import (  # noqa: E402
    RunningRewardScaler as VendorScaler,
)

GAMMA_ENV = 0.99  # == RLSystemContract.gamma_env (contracts.py:31), single source

CFG = RewardScaleConfig(
    method="official_running_return_std",
    gamma=GAMMA_ENV, epsilon=1e-8, cliprew=10.0,
)


def _scaler(cfg=CFG, num_envs=2):
    return RunningRewardScaler(cfg, num_envs=num_envs, gamma_env=GAMMA_ENV)


def _vendor(num_envs=2):
    return VendorScaler(num_envs, cliprew=CFG.cliprew, gamma=CFG.gamma,
                        epsilon=CFG.epsilon, per_env=False)


def _derive_first_documented(episode_ids, prev_last):
    """The documented first-flag rule (test-side independent copy):

    first[e, 0]  = prev_last[e] is None or episode_ids[e, 0] != prev_last[e]
    first[e, t>0] = episode_ids[e, t] != episode_ids[e, t-1]
    """
    n_envs, tlen = episode_ids.shape
    first = np.zeros((n_envs, tlen), dtype=np.int64)
    for e in range(n_envs):
        for t in range(tlen):
            if t == 0:
                first[e, t] = 1 if (prev_last[e] is None
                                    or episode_ids[e, t] != prev_last[e]) else 0
            else:
                first[e, t] = 1 if episode_ids[e, t] != episode_ids[e, t - 1] else 0
    return first


def _rng():
    return np.random.RandomState(0)


# ═══════════════════════════════════════════════════════════════════
# §11.7.4 item 1: step-by-step vendor parity (multi-batch fixed sequence)
# ═══════════════════════════════════════════════════════════════════

def test_scaler_math_parity_with_vendor_fixed_sequence():
    """3 batches, 2 slots: production scaled output, carried ret, and
    running mean/var/count == frozen vendor, exact (same float64
    arithmetic, no tolerance). First flags derived from episode_ids must
    equal the documented rule (this is what the vendor receives from
    rollout done flags)."""
    prod = _scaler()
    vendor = _vendor()
    rng = _rng()
    prev_last = [None, None]
    for batch in range(3):
        # rewards: distinct values per slot so slot mixing is visible
        reward = rng.uniform(-3.0, 3.0, size=(2, 5)).astype(np.float64)
        episode_ids = np.array([
            [10 + batch * 10, 10 + batch * 10, 11 + batch * 10, 11 + batch * 10, 11 + batch * 10],
            [20 + batch * 10] * 5,
        ])
        expected_first = _derive_first_documented(episode_ids, prev_last)
        assert np.array_equal(prod.derive_first(episode_ids), expected_first), (
            f"derive_first deviates from documented rule at batch {batch}"
        )
        scaled_prod = prod(reward, episode_ids)
        scaled_vendor = vendor(reward=reward, first=expected_first)
        assert np.array_equal(scaled_prod, scaled_vendor), (
            f"scaled reward deviates from vendor at batch {batch}"
        )
        params = prod.state_dict()
        assert np.array_equal(np.asarray(params["ret"]), vendor.ret)
        assert params["mean"] == vendor.ret_rms.mean
        assert params["var"] == vendor.ret_rms.var
        assert params["count"] == vendor.ret_rms.count
        assert params["step_count"] == batch + 1
        prev_last = [episode_ids[0, -1], episode_ids[1, -1]]
    # sequence moved: variance is now data-driven, not the init 1.0
    assert prod.state_dict()["var"] != 1.0
    assert prod.state_dict()["count"] != 1e-4


def test_scale_flat_equals_2d_interface():
    """The production flat entry (per-transition slot ids) reproduces the
    2D vendor-parity interface exactly: slot-major flattening preserves
    each slot's time order, so the scaled rewards, the carried per-slot
    returns and the running moments are bitwise identical. This pins the
    flat production path (the one ppo_update consumes) to the
    vendor-pinned 2D path."""
    rng = _rng()
    for batch in range(3):
        prod2d = _scaler()
        prodflat = _scaler()
        reward = rng.uniform(-3.0, 3.0, size=(2, 5)).astype(np.float64)
        episode_ids = np.array([
            [10 + batch * 10, 10 + batch * 10,
             11 + batch * 10, 11 + batch * 10, 11 + batch * 10],
            [20 + batch * 10] * 5,
        ])
        scaled2d = prod2d(reward, episode_ids)
        # slot-major flatten: slot 0's 5 transitions, then slot 1's
        flat_r = reward.reshape(-1)
        flat_ids = episode_ids.reshape(-1)
        flat_env = np.repeat(np.arange(2), 5)
        scaled_flat = prodflat.scale_flat(flat_r, flat_ids, flat_env)
        assert np.array_equal(scaled_flat, scaled2d.reshape(-1)), (
            f"scale_flat deviates from the 2D path at batch {batch}"
        )
        s2 = prod2d.state_dict()
        sf = prodflat.state_dict()
        for key in ("ret", "mean", "var", "count", "step_count"):
            assert s2[key] == sf[key], f"{key} differs on the flat path"
        assert s2["last_episode_id"] == sf["last_episode_id"]


# ═══════════════════════════════════════════════════════════════════
# §11.7.3: episode boundaries — per-slot reset isolation + continuation
# ═══════════════════════════════════════════════════════════════════

def test_scaler_episode_boundary_resets_slot_and_isolates_slots():
    """§11.7.3: a mid-batch new episode (first=1) drops the carried
    return for that slot (ret[t] = reward[t]); the other slot's carried
    return continues across the batch boundary (first=0 at t=0 when the
    episode id is unchanged). Hand-computed recurrence, not vendor."""
    prod = _scaler()
    g = CFG.gamma
    # batch 1: slot 0 switches episode mid-batch (10 -> 11 at t=2);
    # slot 1 stays in one episode; both start fresh (prev_last=None)
    reward = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    episode_ids = np.array([[10, 10, 11, 11], [20, 20, 20, 20]])
    # first flags BEFORE the batch advances the state (t=0 fresh for both)
    assert np.array_equal(prod.derive_first(episode_ids),
                          [[1, 0, 1, 0], [1, 0, 0, 0]])
    prod(reward, episode_ids)
    # batch 1 slot 0: [1, 2+g*1, 3, 4+g*3]   (reset at t=2)
    # batch 1 slot 1: [5, 6+g*5, 7+g*(6+g*5), 8+g*(7+g*(6+g*5))]
    p = prod.state_dict()
    ret0_expected = [1.0, 2.0 + g * 1.0, 3.0, 4.0 + g * 3.0]
    ret1 = 5.0
    ret1 = 6.0 + g * ret1
    ret1 = 7.0 + g * ret1
    ret1 = 8.0 + g * ret1
    assert np.array_equal(np.asarray(p["ret"]), [ret0_expected[-1], ret1])
    # batch 2: slot 0 new episode (12) -> resets at t=0; slot 1 SAME
    # episode (20) -> first=0 at t=0, carried ret1 continues
    reward2 = np.ones((2, 4))
    episode_ids2 = np.array([[12, 12, 12, 12], [20, 20, 20, 20]])
    assert np.array_equal(prod.derive_first(episode_ids2),
                          [[1, 0, 0, 0], [0, 0, 0, 0]])
    prod(reward2, episode_ids2)
    p2 = prod.state_dict()
    # carried ret = LAST timestep: slot 1 continued the batch-1 return
    # (first=0 at t=0) through 4 steps of the new batch
    assert p2["ret"][1] == 1.0 + g * (1.0 + g * (1.0 + g * (1.0 + g * ret1)))
    # slot 0 restarted from zero: t=0 first=1 dropped the batch-1 return,
    # no residue — the recurrence begins at 1.0, not ret0_expected[-1]
    assert p2["ret"][0] == 1.0 + g * (1.0 + g * (1.0 + g * 1.0))


# ═══════════════════════════════════════════════════════════════════
# §11.7.3: none branch — identity transform, structured state
# ═══════════════════════════════════════════════════════════════════

def test_scaler_none_branch_identity_and_structured_state():
    """none: scaled reward == raw reward bitwise; state is structured
    (method/version/step_count) — never an empty dict; step_count
    advances (cross-field uses it); a resumed none scaler keeps the
    identity."""
    cfg = RewardScaleConfig(method="none")
    scaler = RunningRewardScaler(cfg, num_envs=2, gamma_env=GAMMA_ENV)
    reward = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    episode_ids = np.array([[1, 1, 2], [1, 1, 1]])
    for _ in range(2):
        scaled = scaler(reward, episode_ids)
        assert np.array_equal(scaled, reward), "none must be identity"
    params = scaler.state_dict()
    assert params == {"state_version": REWARD_SCALE_STATE_VERSION,
                      "step_count": 2}, "none state must be exactly structured"
    # resume keeps identity and continues step_count
    fresh = RunningRewardScaler(cfg, num_envs=2, gamma_env=GAMMA_ENV)
    fresh.load_state_dict(params)
    scaled = fresh(reward, episode_ids)
    assert np.array_equal(scaled, reward)
    assert fresh.state_dict()["step_count"] == 3


# ═══════════════════════════════════════════════════════════════════
# §11.7.4 item 6: cold resume — next scaled rewards AND state identical
# ═══════════════════════════════════════════════════════════════════

def test_scaler_cold_resume_continues_identically():
    """Cold resume: a loaded scaler's next-batch scaled rewards and
    resulting state == the uninterrupted oracle, exactly."""
    rng = _rng()
    seq = []
    for _ in range(3):
        reward = rng.uniform(-2.0, 2.0, size=(2, 5)).astype(np.float64)
        episode_ids = np.array([
            [0, 0, 1, 1, 1], [2, 3, 3, 3, 3],
        ])
        seq.append((reward, episode_ids))

    oracle = _scaler()
    for reward, episode_ids in seq:
        oracle(reward, episode_ids)
    oracle_params = oracle.state_dict()

    resumed = _scaler()
    resumed.load_state_dict(oracle_params)
    assert resumed.state_dict() == oracle_params

    reward_next = rng.uniform(-2.0, 2.0, size=(2, 5)).astype(np.float64)
    episode_ids_next = np.array([[4, 4, 4, 5, 5], [3, 3, 3, 3, 3]])
    s1 = resumed(reward_next, episode_ids_next)
    s2 = oracle(reward_next, episode_ids_next)
    assert np.array_equal(s1, s2), "resumed next batch must equal oracle"
    assert resumed.state_dict() == oracle.state_dict()


# ═══════════════════════════════════════════════════════════════════
# §11.7.4 item 7: tamper any single axis -> validation fails pre-load
# ═══════════════════════════════════════════════════════════════════

def _two_batch_data(num_envs=2):
    """The fixed 2-batch sequence the running-state fixtures share (the
    state fixtures and their oracles must consume IDENTICAL data)."""
    reward = np.tile(np.arange(1, 6, dtype=np.float64), (num_envs, 1))
    episode_ids = np.arange(num_envs * 5).reshape(num_envs, 5) // 5 + 1
    return reward, episode_ids


def _running_state_dict(num_envs=2, ppo_update_count=3):
    scaler = _scaler(num_envs=num_envs)
    reward, episode_ids = _two_batch_data(num_envs=num_envs)
    for _ in range(ppo_update_count):
        scaler(reward, episode_ids)
    return RewardNormState(
        version="v1.3", method=CFG.method, params=scaler.state_dict(),
    ).to_dict()


def test_validate_reward_scale_state_tamper_each_axis_rejected():
    """method/source/state_version/step_count/ret/mean/var/count/
    last_episode_id/key-set — every single-field mutation fails
    validation (resume-side check, before any tensor load)."""
    state = _running_state_dict(ppo_update_count=3)
    validate_reward_scale_state(state, CFG, ppo_update_count=3)  # valid passes
    mutations = [
        ("method", "none"),  # method swapped vs config
        ("source", "trainer"),  # pinned provider swapped
        ("params.state_version", "v0"),  # state schema swapped
        ("params.step_count", 4),  # cross-field: == ppo_update_count
        ("params.step_count", 2),  # cross-field under-count
        ("params.ret", [np.inf, 0.0]),  # non-finite running return
        ("params.ret", [np.nan, 0.0]),  # non-finite running return
        ("params.mean", np.inf),  # non-finite rms mean
        ("params.var", -0.5),  # negative variance
        ("params.var", np.nan),  # non-finite variance
        ("params.count", -1.0),  # negative count
        ("params.last_episode_id", [1, "x"]),  # non-int episode id
        ("params.last_episode_id", [1]),  # wrong per-slot length
        ("params", {"state_version": "v1"}),  # dropped running axes
        ("params.evil", 1),  # injected extra axis
    ]
    for path, value in mutations:
        bad = copy.deepcopy(state)
        secs = path.split(".")
        if len(secs) == 1:
            bad[secs[0]] = value
        elif len(secs) == 2:
            bad[secs[0]][secs[1]] = value
        with pytest.raises((ValueError, KeyError, AssertionError),
                           match="scale"):
            validate_reward_scale_state(bad, CFG, ppo_update_count=3)


def test_validate_reward_scale_state_none_schema_strict():
    """none state accepts exactly {state_version, step_count} — running
    axes are a schema violation (method-conditional)."""
    good = RewardNormState(
        version="v1.3", method="none",
        params={"state_version": REWARD_SCALE_STATE_VERSION, "step_count": 3},
    ).to_dict()
    validate_reward_scale_state(good, RewardScaleConfig(method="none"),
                                ppo_update_count=3)
    bad = copy.deepcopy(good)
    bad["params"]["ret"] = [0.0, 0.0]  # running-only axis in none state
    with pytest.raises((ValueError, KeyError), match="scale"):
        validate_reward_scale_state(bad, RewardScaleConfig(method="none"),
                                    ppo_update_count=3)
    bad2 = copy.deepcopy(good)
    bad2["params"] = {"state_version": REWARD_SCALE_STATE_VERSION}  # no step_count
    with pytest.raises((ValueError, KeyError), match="scale"):
        validate_reward_scale_state(bad2, RewardScaleConfig(method="none"),
                                    ppo_update_count=3)


# ═══════════════════════════════════════════════════════════════════
# §11.7.3: gamma single-source — mismatch with contract is a hard error
# ═══════════════════════════════════════════════════════════════════

def test_scaler_gamma_mismatch_contract_hard_fail():
    """scaler gamma must equal contract.gamma_env (single-source
    discount). A config with any other gamma is a hard construction
    error for the running method — never a silent scaling mismatch.
    The none branch is exempt (gamma unused)."""
    cfg = RewardScaleConfig(method="official_running_return_std",
                            gamma=0.98)
    with pytest.raises((ValueError, AssertionError), match="gamma"):
        RunningRewardScaler(cfg, num_envs=2, gamma_env=GAMMA_ENV)
    # exact match constructs fine
    RunningRewardScaler(CFG, num_envs=2, gamma_env=GAMMA_ENV)
    # none ignores gamma entirely
    RunningRewardScaler(RewardScaleConfig(method="none", gamma=0.5),
                        num_envs=2, gamma_env=GAMMA_ENV)


def test_reward_scale_config_validation():
    """Malformed configs are config errors, not silent runtime
    behavior: method enum, gamma in (0, 1), positive epsilon/cliprew."""
    with pytest.raises((ValueError, AssertionError)):
        RewardScaleConfig(method="not_a_method")
    with pytest.raises((ValueError, AssertionError)):
        RewardScaleConfig(method="official_running_return_std", gamma=0.0)
    with pytest.raises((ValueError, AssertionError)):
        RewardScaleConfig(method="official_running_return_std", gamma=1.0)
    with pytest.raises((ValueError, AssertionError)):
        RewardScaleConfig(method="official_running_return_std", gamma=-0.1)
    with pytest.raises((ValueError, AssertionError)):
        RewardScaleConfig(method="official_running_return_std", epsilon=0.0)
    with pytest.raises((ValueError, AssertionError)):
        RewardScaleConfig(method="official_running_return_std", cliprew=-1.0)


# ═══════════════════════════════════════════════════════════════════
# checkpoint carrier: RewardNormState roundtrip via the checkpoint form
# ═══════════════════════════════════════════════════════════════════

def test_reward_norm_state_checkpoint_roundtrip():
    """RewardNormState.to_dict() (checkpoint §4.16 shape) -> validate ->
    load -> next batch identical to the oracle; the none variant keeps
    the legacy {version, source, method, params={}} identity shape."""
    state = _running_state_dict(ppo_update_count=2)
    assert state["source"] == "env" and state["method"] == CFG.method
    assert state["version"] == "v1.3"
    validate_reward_scale_state(state, CFG, ppo_update_count=2)

    resumed = _scaler()
    load_scaler_state(state, resumed)
    assert resumed.state_dict()["step_count"] == 2

    # oracle consumes the SAME first 2 batches the loaded state came
    # from, then a fresh third batch — resumed must match it exactly
    reward0, episode_ids0 = _two_batch_data()
    oracle = _scaler()
    for _ in range(2):
        oracle(reward0, episode_ids0)
    reward = np.array([[0.5, 1.5, 2.5], [-1.0, 0.0, 1.0]])
    episode_ids = np.array([[7, 7, 7], [8, 9, 9]])
    s1 = resumed(reward, episode_ids)
    s2 = oracle(reward, episode_ids)
    assert np.array_equal(s1, s2)

    # none variant: params == {} (legacy identity), validate + load fine
    none_state = RewardNormState(
        version="v1.3", method="none",
        params={"state_version": REWARD_SCALE_STATE_VERSION, "step_count": 1},
    ).to_dict()
    validate_reward_scale_state(none_state, RewardScaleConfig(method="none"),
                                ppo_update_count=1)
    none_scaler = RunningRewardScaler(RewardScaleConfig(method="none"),
                                      num_envs=2, gamma_env=GAMMA_ENV)
    load_scaler_state(none_state, none_scaler)
    assert none_scaler.state_dict()["step_count"] == 1
