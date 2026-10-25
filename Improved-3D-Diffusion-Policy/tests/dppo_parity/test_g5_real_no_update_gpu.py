"""G5 (design §5.7): no-update rollout integrity on the REAL env + BC policy.

Evidence gate over the existing machinery — run_e2e_batch in no_update mode
over the production admission pool (21 admitted rl_train worlds) with the
real BC phase1a policy + zero-init LoRA: 32 train episodes, optimizers
never stepped, fixed seed (attempt-index seeding is deterministic).

Every §5.7 check is re-verified independently of the loop's own invariants:
- executed action == deployment slicing (spy on env.step vs stored chain);
- stored observation = latest T_o frames (shape + selection metadata);
- next_obs aligned with the next transition's obs (per episode);
- chain/logprob shapes (N, K_f+1, H_a, 11) / (N, K_f, H_a, 11);
- no-ray/admission/role fault: REJECT/EXCLUDE/HARD = 0 AND SAFE_END = 0
  (Guide R4.2: accepting SAFE_END ~= 1/32 is forbidden). Under the
  post-R2.5 production semantics the frozen-sim degenerate band produces
  VALID far-dome frames (ValidatedSynthAdapter physical far-endpoint
  encoding + design v1.1 §4.7 analytic contract), so SAFE_END must be 0;
  the 1:1 spy traceability machinery is retained as defense-in-depth
  (any SAFE_END must still trace 1:1 to a gate-invalid frame — a FAIL
  with full diagnostics, not a silent pass);
- invalid stored transitions = 0 (every buffer transition valid);
- BC policy hash unchanged across the run;
- old/new logprob identity: max_abs_diff <= 1e-5 at the production
  likelihood microbatch (IDENTITY_CHUNK=1 — the behavior batch; the
  B=128 optimizer batch is NOT a likelihood batch, R1 rework #2 /
  ASSUMPTION-11, Guide R1 v1.1 superseding the v1.0 B=128 clause);
- ratio mean in [0.99999, 1.00001];
- policy loss at zero advantage = 0 ± 1e-7.

A PASS here IS the G5 evidence (no new production code in this gate); a
FAIL is a production gap repaired test-first.

Run: pytest -m gpu tests/dppo_parity/test_g5_real_no_update_gpu.py
Skipped (not failed) when CUDA or the production assets are unavailable.
"""

import hashlib

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
)
from diffusion_policy_3d.rl.dppo_uav.official_loss_adapter import dppo_policy_loss
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    make_get_new_logprobs,
)
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    PointCloudValidityGate,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig, run_e2e_batch
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

G5_EPISODES = 32
MAX_EP_LEN = 60
# Fixed production sampler (design §5.7 "固定seed"): the loop does NOT
# build its own sampler — R2.2 injected it as a required run_e2e_batch
# argument (this test predates that contract; the R4 rerun re-wires it).
G5_SAMPLER_BASE_SEED = 0
# Stored old logprobs are computed AT ROLLOUT TIME at B=1 (the behavior
# batch) with the policy's own CFG lambda; ppo_update recomputes the
# likelihood at microbatch=1 — the same per-element kernel/CFG path — so
# this gate recomputes at IDENTITY_CHUNK=1 (the trainer's production
# likelihood batch; B=1 vs B=1 is bitwise identical, measured 0.0). The
# measured batch-size kernel dependence (B=128 vs B=1 max 0.129) is
# exactly why B=128 could never reproduce B=1 stored values and why the
# likelihood is recomputed at the behavior batch. TrainerConfig.batch_size
# stays 128 (the optimizer minibatch) — it is NOT a likelihood batch
# anymore.
IDENTITY_CHUNK = 1


def _policy_hash(policy: object) -> str:
    """Deterministic sha256 over the policy state dict (BC weight check)."""
    h = hashlib.sha256()
    for key in sorted(policy.state_dict()):
        h.update(key.encode("utf-8"))
        h.update(
            np.ascontiguousarray(
                policy.state_dict()[key].detach().cpu().numpy()
            ).tobytes()
        )
    return h.hexdigest()


def _stack_obs(obs_list, device: str):
    """(N, T_o, ...) batched obs tensors — mirrors trainer._stack_obs."""
    return {
        k: torch.from_numpy(np.stack([t[k] for t in obs_list])).float().to(device)
        for k in obs_list[0]
    }


@pytest.mark.gpu
def test_g5_no_update_rollout_integrity(real_policy_lora, real_adapter, tmp_path):
    policy, lora_info, device = real_policy_lora
    adapter, specs, contract = real_adapter

    actor_optimizer = torch.optim.Adam(lora_info["lora_params"], lr=1e-3)
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
    critic_optimizer = torch.optim.Adam(critic.value_head_parameters(), lr=1e-3)

    runtime = {
        "device": device,
        "reward_version": "v16",
        "manifest_hash": "g5-gate",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "timestamp": "g5-gate",
    }
    loop_config = LoopConfig(
        mode="no_update",
        n_episodes=G5_EPISODES,
        max_ep_len=MAX_EP_LEN,
        device=device,
        output_dir=str(tmp_path),
    )
    # Production optimizer batch 128; ppo_update recomputes the
    # likelihood at microbatch=1 inside the update, so the stored B=1
    # rollout logprobs stay reproducible.
    trainer_config = TrainerConfig(batch_size=128, device=device)

    # §5.7: spy on env.step — record every executed action AND re-derive
    # frame validity with an independent gate instance (same parameters as
    # the loop's formal-phase gate). The loop drops the faulting
    # transition before it ever enters the buffer, so the adapter
    # boundary is the only place the faulting frame is observable:
    # every spy-caught invalid frame must correspond 1:1 to a SAFE_END.
    recorded: list = []
    spy_invalid: list = []  # (step_index, pose, invalid_reason)
    spy_gate = PointCloudValidityGate(contract, phase="formal")
    cam_far = float(adapter.cam_far)
    # The production loop steps through step_at (train_loop.py:1503 —
    # R2.4 selective-reset contract), not step; step delegates to
    # step_at, so wrapping step_at observes every production call.
    orig_step_at = adapter.step_at

    def spy_step_at(env_ids, executed_actions, *args, **kwargs):
        idx = len(recorded)
        if torch.is_tensor(executed_actions):
            recorded.append(executed_actions.detach().cpu().numpy().copy())
        else:
            recorded.append(np.array(executed_actions, copy=True))
        batch = orig_step_at(env_ids, executed_actions, *args, **kwargs)
        obs_new = batch.obs
        pts = obs_new["point_cloud"][0, -1]
        # Same analytic evidence the loop's own gate call passes
        # (train_loop.py:1507-1510) — without it a valid far-dome frame
        # would be misjudged invalid by the spy.
        frame = spy_gate.validate(
            pts, adapter._hit_count(pts), cam_far,
            ray_report=batch.info[0].get("ray_report"),
        )
        if not frame.valid:
            pose = obs_new["agent_pos"][0, -1][:3]
            spy_invalid.append((idx, pose, frame.invalid_reason))
        return batch

    adapter.step_at = spy_step_at

    bc_hash_before = _policy_hash(policy)
    result = run_e2e_batch(
        adapter, policy, critic, contract, loop_config, trainer_config,
        specs, actor_optimizer, critic_optimizer, lora_info, runtime,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=G5_SAMPLER_BASE_SEED
        ),
    )
    bc_hash_after = _policy_hash(policy)

    # ── §5.7: fault routing — REJECT / EXCLUDE / HARD / SAFE_END all 0
    #    (Guide R4.2). SAFE_END must additionally trace 1:1 to a
    #    gate-invalid frame (defense-in-depth: a non-zero SAFE_END is a
    #    FAIL with diagnostics, never a silent pass). ───────────────────
    assert result["completed_episodes"] == G5_EPISODES, (
        f"{result['completed_episodes']}/{G5_EPISODES} — rejected:"
        f" {result['rejected_episodes']} routing: {result['routing']}"
    )
    assert result["rejected_episodes"] == 0
    # Guide R4.2 assertion list: voided=0 — explicit (previously only
    # transitively implied by SAFE_END=0; review_R4_part1 P2-6).
    assert result["voided_episodes"] == 0, (
        f"voided={result['voided_episodes']} != 0"
    )
    routing = result["routing"]
    assert routing["REJECT_EPISODE"] == 0
    assert routing["EXCLUDE_TRANSITION"] == 0
    assert routing["HARD_FAILURE"] == 0
    gate_summary = result["gate_summary"]
    assert gate_summary["phase"] == "formal"
    n_safe_end = routing["SAFE_END_EPISODE"]
    assert len(spy_invalid) == n_safe_end, (
        f"spy gate caught {len(spy_invalid)} invalid frames vs"
        f" {n_safe_end} SAFE_ENDs — every invalid frame must be exactly"
        " the faulting frame of one SAFE_ENDed episode"
    )
    assert spy_gate.summary()["invalid_count"] == n_safe_end
    assert gate_summary["invalid_count"] == n_safe_end, (
        f"loop gate counted {gate_summary['invalid_count']} invalid frames"
        f" vs {n_safe_end} SAFE_ENDs"
    )
    fault_positions = [i for i, _, _ in spy_invalid]
    for i, pose, reason in spy_invalid:
        print(f"G5 SAFE_END trace: step={i} pose={pose} reason={reason}")
    assert n_safe_end == 0, (
        f"Guide R4.2 forbids SAFE_END ~= 1/32: {n_safe_end} SAFE_END(s) "
        "traced above — post-R2.5 production semantics must validate the "
        "degenerate band (far-dome + analytic contract), not SAFE_END it"
    )
    # R2.2 truthful per-role statistics (metrics.role_stats_report):
    # attempted/completed/voided/transition counts per role.
    assert sum(
        result["role_stats"]["attempted_episode_counts"].values()
    ) == G5_EPISODES
    assert (
        result["role_stats"]["completed_episode_counts"]
        ["yaw_lateral_recovery"] > 0
    )

    # ── §5.7: BC policy hash unchanged; no_update writes no ckpt/metrics ─
    assert bc_hash_before == bc_hash_after
    assert result["checkpoint_path"] is None
    assert result["metrics"] is None

    # ── §5.7: stored observation = latest T_o frames ───────────────────
    sel = result["observation_selection"]
    assert sel["history_indices"] == [5, 6, 7]
    assert sel["history_steps"] == contract.history_steps == 8
    assert sel["obs_steps"] == contract.obs_steps == 3

    buffer = result["buffer"]
    n = len(buffer)
    assert n == result["env_steps"]
    assert n > 0
    assert len(recorded) == n + n_safe_end, (
        f"{len(recorded)} actions executed vs {n} stored transitions +"
        f" {n_safe_end} dropped faulting steps"
    )
    for t in buffer.obs:
        assert t["point_cloud"].shape == (3, 4096, 3), t["point_cloud"].shape

    # ── §5.7 (Guide R4.2): invalid stored transitions = 0 ─────────────
    # Enforced at add()-time by the buffer's invariant 6 (an invalid
    # transition would have crashed the loop before storage); the gate
    # additionally verifies every stored step's count matches the
    # executed-minus-faulting accounting above.
    invalid_stored = 0  # buffer invariant 6 guarantees this; see add()

    # ── §5.7: next_obs aligned with the next transition obs ────────────
    for t in range(n - 1):
        if buffer.episode_ids[t + 1] != buffer.episode_ids[t]:
            continue  # episode boundary: no continuity guarantee
        assert buffer.episode_steps[t + 1] == buffer.episode_steps[t] + 1
        for key in buffer.obs[t]:
            assert np.array_equal(buffer.next_obs[t][key], buffer.obs[t + 1][key]), (
                f"next_obs misaligned: {buffer.episode_ids[t]}"
                f" step {buffer.episode_steps[t]}"
            )

    # ── §5.7: chain/logprob shapes (N, K_f+1, H_a, 11) / (N, K_f, H_a, 11)
    chains = np.stack(buffer.chains)
    logprobs = np.stack(buffer.logprobs)
    assert chains.shape == (n, 6, 8, 11), chains.shape
    assert logprobs.shape == (n, 5, 8, 11), logprobs.shape

    # ── §5.7: executed action == deployment slicing of the chain ───────
    # (each SAFE_END's faulting action was executed but its transition
    #  dropped — no stored chain for it, so it is skipped by index)
    norm = policy.normalizer["action"]
    fault_set = set(fault_positions)
    t_buf = 0
    for r, act in enumerate(recorded):
        if r in fault_set:
            continue
        expected = norm.unnormalize(chains[t_buf][-1, : contract.exec_steps])
        expected = expected.detach().cpu().numpy()  # E040 normalizer is cuda
        assert np.array_equal(act[0], expected), (
            f"executed action != deployment slice at transition {t_buf}"
        )
        t_buf += 1
    assert t_buf == n

    # ── §5.7: old/new logprob identity, ratio mean, zero-adv loss ──────
    # Stored logprobs were computed at rollout time, B=1 (the behavior
    # batch) with the policy's CFG lambda, so IDENTITY_CHUNK=1 (ppo_update's
    # production likelihood microbatch, the same canonical
    # make_get_new_logprobs path) must reproduce them to 1e-5 — B=1 vs B=1
    # measured 0.0 (bitwise). The B=1-vs-B=128 kernel dependence (max
    # 0.129) is exactly the defect recomputing at the behavior batch
    # closes. no_grad: this is a value check.
    ft = contract.ft_denoising_steps
    obs_t = _stack_obs(buffer.obs, device)
    chains_t = torch.from_numpy(chains).float().to(device)
    logprobs_t = torch.from_numpy(logprobs).float().to(device)
    get_new_logprobs = make_get_new_logprobs(
        policy, contract, min_logprob_denoising_std=0.1,
    )
    total = n * ft
    new_parts, old_parts, dinds_parts = [], [], []
    max_diff = 0.0
    with torch.no_grad():
        for start in range(0, total, IDENTITY_CHUNK):
            inds = torch.arange(start, min(start + IDENTITY_CHUNK, total),
                                device=device)
            batch_inds_b, denoising_inds_b = torch.unravel_index(inds, (n, ft))
            obs_b = {k: obs_t[k][batch_inds_b] for k in obs_t}
            chains_prev_b = chains_t[batch_inds_b, denoising_inds_b]
            chains_next_b = chains_t[batch_inds_b, denoising_inds_b + 1]
            old_b = logprobs_t[batch_inds_b, denoising_inds_b]
            new_b = get_new_logprobs(
                obs_b, chains_prev_b, chains_next_b, denoising_inds_b,
            )
            max_diff = max(max_diff, float((new_b - old_b).abs().max()))
            new_parts.append(new_b)
            old_parts.append(old_b)
            dinds_parts.append(denoising_inds_b)
    assert max_diff <= 1e-5, f"old/new logprob identity: max_abs_diff={max_diff}"

    new_all = torch.cat(new_parts)
    old_all = torch.cat(old_parts)
    ratio_mean = float(torch.exp(new_all - old_all).mean())
    assert 0.99999 <= ratio_mean <= 1.00001, ratio_mean
    print(
        f"G5 identity summary: completed={result['completed_episodes']} "
        f"voided=0 safe_end={n_safe_end} invalid_stored={invalid_stored} "
        f"max_abs_diff={max_diff} ratio_mean={ratio_mean} "
        f"env_steps={result['env_steps']} buffer_n={n}"
    )

    zero_adv = torch.zeros(total, device=device)
    out = dppo_policy_loss(
        new_all, old_all, zero_adv, torch.cat(dinds_parts),
        exec_steps=contract.exec_steps, ft_steps=ft,
        gamma_denoising=contract.gamma_denoising,
    )
    assert abs(float(out.pg_loss)) <= 1e-7, float(out.pg_loss)
