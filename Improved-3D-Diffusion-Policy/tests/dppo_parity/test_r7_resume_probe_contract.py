"""F4 closure guide §3 (Task 2): probe contract RED tests + the formal
active-slot contract.

The seven tests replace the rejected draft
(06_resume/rootcause_probe9/rejected_drafts/test_active_slot_states_contract_r7.py)
whose premise — episode-mode checkpoints must carry all n_envs slots —
is REFUTED by the quiescent semantics (train_loop.py:2568 gates
active_slot_states filling on ``target is not None``):

  #1-#5  the regression-grade probe (scripts/r7_resume_rootcause_probe.py)
         contract: step_at-not-step wiring, per-update capture with
         explicit clearing, REAL rollout result fields with hard-fail on
         missing fields, observation component split, and no zip-truncate;
  #6     episode-mode batch ends with active_slot_states == {} (the
         quiescent contract the F4 diagnosis depends on);
  #7     a transition-mode mid-episode cut SAVES the live slot
         (non-empty, done=False) and a COLD resume (fresh env, from the
         checkpoint) continues with next observation / task /
         episode_step / slot RNG exactly equal to the in-process
         continuation.

Tests 1-5 are RED until the probe module exists; 6-7 document the
production contract and must stay green.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import r7_resume_rootcause_probe as probe_mod  # noqa: E402

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract  # noqa: E402
from diffusion_policy_3d.rl.dppo_uav.critic import (  # noqa: E402
    PolicyObsEncoder,
    ValueCritic,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (  # noqa: E402
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (  # noqa: E402
    LoopConfig,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig  # noqa: E402

from fake_policy import FakePolicy  # noqa: E402
from mock_uav_env import MockUAVEnv  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# shared fakes for the probe contract tests (CPU, no training loop)
# ═══════════════════════════════════════════════════════════════════

def _step_result():
    return SimpleNamespace(
        reward=np.array([1.0]),
        terminated=np.array([False]),
        truncated=np.array([False]),
        info=[],
    )


def _rollout_result():
    """A result object carrying the REAL sample_dppo_rollout fields
    (phase1a_actor_adapter.py:280-294) — chains_policy / old_logprobs /
    executed_action. ``chain`` does NOT exist on DPPORolloutSample."""
    return SimpleNamespace(
        chains_policy=torch.randn(4, 8, 11),
        old_logprobs=torch.randn(4, 8, 11),
        executed_action=torch.randn(1, 11, 3),
    )


class _FakeEnv:
    """Minimal per-slot env surface the probe records (the kinematic
    uav env attribute names, code-verified by probe 9 on the real env)."""

    n_envs = 1

    def __init__(self):
        self._drone_pos = np.zeros(3)
        self._drone_yaw = 0.0
        self._drone_vel = np.zeros(3)
        self.episode_step = 0
        self._disturb_idx = 0
        self._count_stuck = 0
        self._prev_dist = 1.0
        self._episode_rng_seed = 42
        self._pc_history = np.zeros((2, 3))
        self._state_history = np.zeros((2, 3))
        self._rng = np.random.RandomState(0)

    def is_active(self, env_id):
        return True

    def slot_record(self, env_id):
        return SimpleNamespace(
            episode_id="fake:anticipatory:000000042",
            episode_seed=42,
            task_uid="fake",
        )

    def slot_spec(self, env_id):
        return SimpleNamespace(world_uid="fake", role="anticipatory")

    def state_dict(self, env_id):
        return {"adapter": {"done": False}, "env": {"rng_state": self._rng.get_state()}}


class _CountedAdapter:
    """Fake adapter distinguishing the eval path (step) from the
    training path (step_at) — guide §3.1 #1's wiring discriminator."""

    def __init__(self, env):
        self._envs = [env]
        self.step_calls = 0
        self.step_at_calls = 0

    def step(self, actions):
        self.step_calls += 1
        return _step_result()

    def step_at(self, env_ids, executed_actions):
        self.step_at_calls += 1
        return _step_result()

    # Real adapters delegate the per-slot env surface to their envs
    # (UAVVectorEnvAdapter.is_active/slot_record/slot_spec/state_dict);
    # the probe contract reads those from the attached adapter, so the
    # fixture must mirror the delegation or record_rollout fails.
    def is_active(self, env_id):
        return self._envs[env_id].is_active(env_id)

    def slot_record(self, env_id):
        return self._envs[env_id].slot_record(env_id)

    def slot_spec(self, env_id):
        return self._envs[env_id].slot_spec(env_id)

    def state_dict(self, env_id):
        return self._envs[env_id].state_dict(env_id)


class _NoStepAt:
    pass


def _make_probe(tmp_path, **kw):
    kw.setdefault("out_path", tmp_path / "probe.json")
    return probe_mod.RootCauseProbe(probe_mod.ProbeConfig(**kw))


def _armed(tmp_path, adapter_cls=_CountedAdapter, **cfg_kw):
    """Probe with step hook installed on the fake adapter class, env
    attached, and (by default) recording armed for update 1."""
    env = _FakeEnv()
    cfg_kw.setdefault("capture_update", 1)
    probe = _make_probe(tmp_path, **cfg_kw)
    probe.install_step_hook(adapter_cls)
    adapter = adapter_cls(env)
    probe.attach_env(adapter)
    probe.begin_update(1)
    return probe, adapter, env


def _policy_and_obs():
    return SimpleNamespace(), {
        "point_cloud": torch.zeros(1, 5, 3),
        "agent_pos": torch.zeros(1, 3),
    }


# ═══════════════════════════════════════════════════════════════════
# #1  step_at-not-step wiring
# ═══════════════════════════════════════════════════════════════════

def test_probe_hooks_step_at_not_step(tmp_path):
    """guide §3.1 #1: the probe's step hook wraps step_at (the training
    path) — never step (the eval path). Record count equals the step_at
    call count and exceeds zero; install on a class without step_at is a
    loud ProbeContractError (fail-closed hook miss, §3.3)."""
    original = _CountedAdapter.step_at  # must capture BEFORE the hook install
    probe, adapter, env = _armed(tmp_path)
    try:
        assert type(adapter).step_at is not original, (
            "attach_env must verify the hook actually resolved")
        adapter.step_at([0], np.zeros((1, 11)))
        adapter.step(np.zeros((1, 11)))
        adapter.step_at([0], np.zeros((1, 11)))
        assert adapter.step_at_calls == 2
        assert adapter.step_calls == 1
        assert probe.step_records == 2, (
            "probe records must equal the step_at call count")
        assert probe.rollout_records == 0, (
            "the eval-path step must not be mistaken for training")
        assert all(s["slot"] == 0 for s in probe.steps)
    finally:
        probe.uninstall()
    assert _CountedAdapter.step_at is original, "uninstall must restore"
    with pytest.raises(probe_mod.ProbeContractError):
        probe_mod.RootCauseProbe(
            probe_mod.ProbeConfig(out_path=tmp_path / "noop.json")
        ).install_step_hook(_NoStepAt)


# ═══════════════════════════════════════════════════════════════════
# #2  per-update capture with explicit clearing
# ═══════════════════════════════════════════════════════════════════

def test_probe_clears_capture_at_requested_update(tmp_path):
    """guide §3.1 #2: capture_update=3 must produce an artifact
    containing ONLY update-3 frames — never the 1..3 concatenation of
    the probe-9 defect (hooks installed before update 1, no clear)."""
    probe, adapter, env = _armed(tmp_path, capture_update=3)
    policy, obs_b = _policy_and_obs()
    try:
        probe.begin_update(1)
        probe.record_rollout(policy, obs_b, _rollout_result())
        adapter.step_at([0], np.zeros((1, 11)))
        probe.begin_update(2)
        probe.record_rollout(policy, obs_b, _rollout_result())
        adapter.step_at([0], np.zeros((1, 11)))
        probe.begin_update(3)
        probe.record_rollout(policy, obs_b, _rollout_result())
        adapter.step_at([0], np.zeros((1, 11)))
        probe.record_rollout(policy, obs_b, _rollout_result())
        adapter.step_at([0], np.zeros((1, 11)))
        rc = probe.finalize()
        assert rc == 0, f"probe finalize failed with {rc}"
    finally:
        probe.uninstall()
    artifact = json.loads((tmp_path / "probe.json").read_text())
    frames = artifact["frames"]
    assert len(frames) == 2, (
        "artifact must contain ONLY update-3 frames, not the 1..3 "
        f"concatenation (got {len(frames)})")
    assert all(f["absolute_update"] == 3 for f in frames)
    assert [f["relative_transition_index"] for f in frames] == [0, 1], (
        "relative indices must reset at the capture update")


# ═══════════════════════════════════════════════════════════════════
# #3  REAL rollout result fields, hard-fail on missing fields
# ═══════════════════════════════════════════════════════════════════

def test_probe_uses_real_rollout_result_fields(tmp_path):
    """guide §3.1 #3: the probe reads res.chains_policy /
    res.old_logprobs / res.executed_action — the REAL DPPORolloutSample
    fields (phase1a_actor_adapter.py:280-294). The non-existent
    res.chain must hard-fail: a missing field is a ProbeContractError
    naming the field, never a recorded None."""
    probe, adapter, env = _armed(tmp_path)
    policy, obs_b = _policy_and_obs()
    try:
        probe.record_rollout(policy, obs_b, _rollout_result())
        assert probe.rollout_records == 1
        frame = probe.frames[-1]
        for key in ("chains_policy_hash", "old_logprobs_hash",
                    "executed_action_hash"):
            assert frame[key], f"{key} must be recorded"
        # probe9 defect: the old probe read the NON-EXISTENT res.chain.
        # The contract check walks ROLLOUT_RESULT_FIELDS in order and
        # hard-fails on the first missing REAL field — so craft a result
        # that carries the fake 'chain' plus every REAL field except
        # executed_action: the error must name executed_action, never
        # read the fake 'chain'.
        fake_chain = SimpleNamespace(
            chain=torch.zeros(1),
            chains_policy=torch.zeros(1, 8, 11),
            old_logprobs=torch.zeros(1, 8, 11),
        )
        with pytest.raises(probe_mod.ProbeContractError) as ei:
            probe.record_rollout(policy, obs_b, fake_chain)
        assert "executed_action" in str(ei.value), (
            "the missing-field error must name the absent REAL field")
    finally:
        probe.uninstall()


# ═══════════════════════════════════════════════════════════════════
# #4  observation component split
# ═══════════════════════════════════════════════════════════════════

def test_probe_splits_observation_components(tmp_path):
    """guide §3.1 #4: obs_pc_hash / obs_agent_hash / obs_combined_hash
    recorded separately; compare_dumps reports the FIRST-diff component
    accurately when only one component changes."""
    policy, _ = _policy_and_obs()
    ag = torch.randn(1, 3)
    pc_a = torch.randn(1, 5, 3)
    pc_b = pc_a + 1.0
    # Every randn (including _rollout_result) is consumed BEFORE any
    # recording — the global torch RNG state is then frozen across the
    # recorded frames, so the RNG hashes stay equal and the compare
    # isolates the observation component.
    r1, r2, r3 = _rollout_result(), _rollout_result(), _rollout_result()
    probe, adapter, env = _armed(tmp_path)
    try:
        probe.record_rollout(policy, {"point_cloud": pc_a, "agent_pos": ag}, r1)
        probe.record_rollout(policy, {"point_cloud": pc_b, "agent_pos": ag}, r2)
        # frame 3 is a pc_a frame again: the resume dump below is
        # [f1(pc_a), f3(pc_a)] vs control [f1(pc_a), f2(pc_b)] — the
        # only differing quantity at index 1 is obs_pc_hash, and
        # FRAME_ORDER walks obs_* before chains_*/logprobs/action.
        probe.record_rollout(policy, {"point_cloud": pc_a, "agent_pos": ag}, r3)
        adapter.step_at([0], np.zeros((1, 11)))
        rc = probe.finalize()
        assert rc == 0
    finally:
        probe.uninstall()
    frames = json.loads((tmp_path / "probe.json").read_text())["frames"]
    f1, f2, f3 = frames
    assert f1["obs_pc_hash"] != f2["obs_pc_hash"], (
        "point-cloud change must move obs_pc_hash")
    assert f1["obs_agent_hash"] == f2["obs_agent_hash"], (
        "agent component unchanged must keep obs_agent_hash")
    assert f1["obs_combined_hash"] != f2["obs_combined_hash"]
    # control = [f1(pc_a), f2(pc_b)], resume = [f1(pc_a), f3(pc_a)] —
    # the only difference at index 1 is the point cloud; first diff
    # must be reported as obs_pc_hash, not a fused obs hash and not
    # a later chains/logprobs field.
    ctrl = {"frames": frames[:2], "steps": []}
    resm = {"frames": [frames[0], frames[2]], "steps": []}
    out = probe_mod.compare_dumps(ctrl, resm)
    assert out["first_diff"] == {"kind": "frame", "index": 1,
                                 "field": "obs_pc_hash"}, out["first_diff"]
    # the other direction: agent change -> obs_agent_hash
    ag_c = ag + 1.0
    r4, r5 = _rollout_result(), _rollout_result()
    probe2, adapter2, env2 = _armed(tmp_path)
    try:
        probe2.record_rollout(policy, {"point_cloud": pc_a, "agent_pos": ag}, r4)
        probe2.record_rollout(policy, {"point_cloud": pc_a, "agent_pos": ag_c}, r5)
        adapter2.step_at([0], np.zeros((1, 11)))
        rc = probe2.finalize()
        assert rc == 0
    finally:
        probe2.uninstall()
    frames2 = json.loads((tmp_path / "probe.json").read_text())["frames"]
    ctrl2 = {"frames": [frames2[0]], "steps": []}
    resm2 = {"frames": [frames2[1]], "steps": []}
    out2 = probe_mod.compare_dumps(ctrl2, resm2)
    assert out2["first_diff"] == {"kind": "frame", "index": 0,
                                  "field": "obs_agent_hash"}, out2["first_diff"]


# ═══════════════════════════════════════════════════════════════════
# #4b  adapter vs inner-env attribute ownership
# ═══════════════════════════════════════════════════════════════════

class _BareEnv:
    """Inner-env surface ONLY (all REQUIRED_ENV_ATTRS) — no is_active /
    slot_record / slot_spec: those live on the adapter
    (environment_adapter.py:1040/1217/1221), and the inner KinematicUAVEnv
    has none of them."""

    n_envs = 1

    def __init__(self):
        self._drone_pos = np.zeros(3)
        self._drone_yaw = 0.0
        self._drone_vel = np.zeros(3)
        self.episode_step = 0
        self._disturb_idx = 0
        self._count_stuck = 0
        self._prev_dist = 1.0
        self._episode_rng_seed = 42
        self._pc_history = np.zeros((2, 3))
        self._state_history = np.zeros((2, 3))
        self._rng = np.random.RandomState(0)


class _AdapterNoDelegate:
    """Adapter with step_at + is_active but NO slot_record/slot_spec —
    the probe must read record/spec defensively (hasattr) and take
    is_active from the adapter, never the inner env."""

    def __init__(self, env):
        self._envs = [env]
        self.step_at_calls = 0

    def step_at(self, env_ids, executed_actions):
        self.step_at_calls += 1
        return _step_result()

    def is_active(self, env_id):
        return True


def test_probe_adapter_surface_not_inner_env(tmp_path):
    """The record_step contract attributes (is_active, slot_record)
    belong to the ADAPTER, not the inner env — a bare inner env must
    record cleanly (probe-9 lesson: attribute-name mismatches on the
    real env surface crash the probe mid-run, burning GPU hours)."""
    env = _BareEnv()
    probe = _make_probe(tmp_path)
    probe.install_step_hook(_AdapterNoDelegate)
    adapter = _AdapterNoDelegate(env)
    probe.attach_env(adapter)
    probe.begin_update(1)
    try:
        adapter.step_at([0], np.zeros((1, 11)))
        assert probe.step_records == 1
        step = probe.steps[-1]
        assert step["done"] is False
        assert step["slot"] == 0
        assert step["episode_step"] == 0
        assert step["slot_rng_hash"]
        assert step["reward"] == 1.0  # _step_result() reward
        rc = probe.finalize()
        assert rc == 0
    finally:
        probe.uninstall()


# ═══════════════════════════════════════════════════════════════════
# #5  never zip-truncate
# ═══════════════════════════════════════════════════════════════════

def test_probe_never_zip_truncates():
    """guide §3.1 #5: unequal control/resume lengths must FAIL explicitly
    and report BOTH lengths — the offset-2213 misalignment of probe 9
    must never silently zip-truncate again."""
    frame = {"obs_pc_hash": "a" * 64}
    ctrl = {"frames": [frame] * 10, "steps": []}
    resm = {"frames": [frame] * 8, "steps": []}
    out = probe_mod.compare_dumps(ctrl, resm)
    assert out["match"] is False
    assert "control" in out["fail"] and "resume" in out["fail"]
    assert "10" in out["fail"] and "8" in out["fail"], out["fail"]


# ═══════════════════════════════════════════════════════════════════
# #6  episode-mode quiescent checkpoint — no active slots
# ═══════════════════════════════════════════════════════════════════

def _specs(contract):
    return [
        EpisodeTaskSpec(
            world_uid=f"mock_probe_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock_probe__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_probe_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]


def _runtime():
    return {
        "device": "cpu",
        "reward_version": "v16",
        "manifest_hash": "probe-contract",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "world_assets_hash": "?",
        "timestamp": "probe-contract",
    }


def test_episode_mode_quiescent_checkpoint_has_no_active_slots(tmp_path):
    """guide §3.1 #6 (the rejected draft's opposite): an episode-mode
    batch completes with active_slot_states == {} AND its checkpoint
    carries none — train_loop.py:2568 only fills active_slot_states
    under ``target is not None`` (transition mode). This quiescence is
    the F4 diagnosis's premise, not a defect."""
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    contract = RLSystemContract()
    policy = FakePolicy(contract)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    cfg = LoopConfig(
        mode="one_update", n_episodes=2, max_ep_len=12, device="cpu",
        output_dir=str(tmp_path),
    )
    res = run_e2e_batch(
        MockUAVEnv(contract), policy, critic, contract, cfg,
        TrainerConfig(batch_size=16, device="cpu"), specs,
        torch.optim.Adam(list(policy.model.parameters()), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        {"lora_params": [], "n_adaptors": 0}, _runtime(),
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )
    assert res["completed_episodes"] == 2
    assert res["active_slot_states"] == {}, (
        "episode mode is quiescent — no active slots at the boundary")
    data = CheckpointBuilder.load(Path(res["checkpoint_path"]))
    assert not data.active_slot_states, (
        "an episode-mode checkpoint must NOT carry active slot states")


# ═══════════════════════════════════════════════════════════════════
# #7  transition-mode cut saves the live slot; cold resume is exact
# ═══════════════════════════════════════════════════════════════════

def _obs_hash(obs: dict) -> str:
    from diffusion_policy_3d.rl.dppo_uav.train_loop import _canonical_bytes

    h = hashlib.sha256()
    for k in sorted(obs):
        h.update(_canonical_bytes(obs[k]))
    return h.hexdigest()


def _transition_batch(tmp_path, env, policy, critic, opt_a, opt_c, specs,
                      runtime, sampler, *, resume_from=None, out_dir):
    cfg = LoopConfig(
        mode="one_update", n_episodes=4, max_ep_len=20, device="cpu",
        output_dir=str(out_dir),
        target_transitions=25, total_transition_budget=50,
        rollout_transitions_per_update=25,
    )
    kwargs = {}
    if resume_from:
        kwargs["resume_from"] = resume_from
    return run_e2e_batch(
        env, policy, critic, RLSystemContract(), cfg,
        TrainerConfig(batch_size=16, device="cpu"), specs,
        opt_a, opt_c, {"lora_params": [], "n_adaptors": 0}, runtime,
        task_sampler=sampler, **kwargs,
    )


def test_transition_cut_checkpoint_restores_live_slot(tmp_path):
    """guide §3.1 #7: a transition-mode cut mid-episode SAVES the active
    slot (non-empty, done=False); a COLD resume — fresh env, from the
    checkpoint — continues with next observation / task / episode_step /
    slot RNG exactly equal to the in-process continuation (the live-slot
    adopt path, train_loop.py:2288-2300)."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    critic = ValueCritic(contract, PolicyObsEncoder(policy))
    specs = _specs(contract)
    runtime = _runtime()
    lora_info = {"lora_params": [], "n_adaptors": 0}

    def _opts():
        return (
            torch.optim.Adam(list(policy.model.parameters()), lr=1e-3),
            torch.optim.Adam(critic.parameters(), lr=1e-3),
        )

    env_a = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    opt_a1, opt_c1 = _opts()
    res_a = _transition_batch(
        tmp_path, env_a, policy, critic, opt_a1, opt_c1, specs, runtime,
        EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0),
        out_dir=tmp_path,
    )
    active = res_a["active_slot_states"]
    assert active, "the transition cut must leave an ACTIVE slot"
    for st in active.values():
        assert st["adapter"]["done"] is False, (
            "the cut slot must stay ACTIVE (not closed)")
    ckpt = res_a["checkpoint_path"]
    assert ckpt

    # COLD resume: fresh env + fresh sampler from the checkpoint
    # (load_state_dict path — the F4 cross-process shape).
    env_b = MockUAVEnv(contract, reach_step=1000, max_steps=1000)
    opt_b1, opt_b2 = _opts()
    res_b = _transition_batch(
        tmp_path, env_b, policy, critic, opt_b1, opt_b2, specs, runtime,
        EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0),
        resume_from=ckpt, out_dir=tmp_path,
    )
    # in-process continuation reference: SAME env (live adopt path)
    opt_c1, opt_c2 = _opts()
    res_c = _transition_batch(
        tmp_path, env_a, policy, critic, opt_c1, opt_c2, specs, runtime,
        EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0),
        resume_from=ckpt, out_dir=tmp_path,
    )
    assert res_b["env_steps"] == res_c["env_steps"] == 25, (
        f"resumed batches must collect the remaining budget "
        f"(b={res_b['env_steps']}, c={res_c['env_steps']})")

    bb, cb = res_b["buffer"], res_c["buffer"]
    assert len(bb.obs) == len(cb.obs), (
        f"cold({len(bb.obs)}) vs in-process({len(cb.obs)}) transitions")
    for i in range(len(bb.obs)):
        assert _obs_hash(bb.obs[i]) == _obs_hash(cb.obs[i]), (
            f"next observation diverges at transition {i}")
        assert bb.episode_ids[i] == cb.episode_ids[i], (
            f"next task diverges at transition {i}")
        assert bb.episode_steps[i] == cb.episode_steps[i], (
            f"episode_step diverges at transition {i}")
        assert bb.rewards[i] == cb.rewards[i], (
            f"reward diverges at transition {i}")
    # slot RNG exact: both resumed envs consumed identical RNG streams
    rb = env_b.state_dict(0)["env"]["rng_state"]
    rc = env_a.state_dict(0)["env"]["rng_state"]
    for x, y in zip(rb, rc):
        if hasattr(x, "shape"):
            assert (x == y).all(), "slot RNG state diverged after resume"
        else:
            assert x == y
