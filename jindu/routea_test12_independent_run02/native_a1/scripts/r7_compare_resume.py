"""r7_compare_resume.py — R7-6C fixed same-seed control-vs-resume comparator.

guide R7-6 Step 6-7:

    control: seed 202, uninterrupted updates 1..5
    resume:  seed 202, updates 1..2 -> save v6 -> NEW PROCESS -> updates 3..5

Every update snapshots the NINE comparison layers (first mismatch stops
the compare):

    1 load-complete state
    2 next task/world/episode seed/start/goal
    3 rollout noise/action/chain/oldlogprob
    4 reward/outcome/buffer/next-values
    5 values/advantages/returns
    6 randperm/minibatch indices
    7 actor/critic grads
    8 parameters/optimizer moments
    9 sampler/slots/counters/metrics

Step 7 exact gate: discrete states/hashes/cursors/counters NEVER use
tolerances; FP tensors compare by exact hash under the formal
deterministic numerical mode (configured in every phase process before
any workload).

Phases (each a separate process — the resume side is a cold start):

    --phase control   5 updates fresh; snapshots -> workdir/control/
    --phase resume-a  2 updates fresh; snapshots -> workdir/resume_a/;
                      manifest.json records the update-2 checkpoint
    --phase resume-b  --resume-from the update-2 checkpoint; updates
                      3..5; snapshots -> workdir/resume_b/
    --phase compare   nine-layer compare, first-diff stop -> summary.json
    --all             control -> resume-a -> resume-b -> compare, each
                      stage in its OWN subprocess

Summary schema (consumed by tests/dppo_parity/test_resume_determinism_r7.py):

    {schema_version, pass, metric_diffs, ckpt_diffs,
     first_diff: {update, layer, control, resume, detail} | null,
     control: {ckpt, updates}, resume: {ckpt, updates},
     layers_compared: [...]}

Run:
    python scripts/r7_compare_resume.py --all --workdir <dir>
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Deterministic-process protocol (production seeding semantics — the
# launcher fixes PYTHONHASHSEED before interpreter start): every phase
# process (spawned below) inherits a FIXED hash seed, so the
# environment_rng_state restorable gate (validate_environment_rng_state)
# passes and the point-cloud RNG is cross-process reproducible. An unset
# seed makes the checkpoint gate hard-fail — that is the fail-closed
# behaviour this comparator must avoid by construction.
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests" / "dppo_parity"
for p in (str(REPO_ROOT), str(TESTS_DIR), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import (
    PolicyObsEncoder,
    ValueCritic,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    FORMAL_NUMERICAL_MODE,
    LoopConfig,
    _canonical_bytes,
    _state_hash,
    configure_numerical_mode,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv

from r7_resume_rootcause_probe import ProbeConfig, RootCauseProbe

SCHEMA_VERSION = "r7-compare-v1"
BASE_SEED = 202
UPDATES = 5
RESUME_SPLIT = 2

# F4 (Zread P1): production assets for --real-gpu — same defaults as
# tests/dppo_parity/conftest.py; env overrides let a mirror point
# elsewhere.  The real mode runs the SAME cross-process protocol as the
# CPU mock, on the production BC policy (E5 retrain, sha256 c1fca4ac…),
# registered LoRA, E040 critic and UAVVectorEnvAdapter, on CUDA.
DEFAULT_BC_CKPT = str(
    REPO_ROOT
    / "data/outputs/2026.08.16/08.53.26_e5_lambda0p5"
    / "checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)

ROLE_ORDER = (
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
)

LAYERS = [
    "1_load_complete_state",
    "2_next_task_world_episode",
    "3_rollout_noise_action_chain_logprob",
    "4_reward_outcome_buffer_next_values",
    "5_values_advantages_returns",
    "6_randperm_minibatch",
    "7_actor_critic_grads",
    "8_parameters_optimizer_moments",
    "9_sampler_slots_counters_metrics",
]

# The update number embedded in every checkpoint filename — the ckpt
# presence/update-number axis of the Step 7 exact gate.
_CKPT_UPD_RE = re.compile(r"_ppoupd_(\d+)\.pt$")


def _spec(world_uid: str, role: str) -> EpisodeTaskSpec:
    return EpisodeTaskSpec(
        world_uid=world_uid,
        obstacle_json="{}",
        role=role,
        split="rl_train",
        admission_id=f"{world_uid}__{role}",
        generator_version="mock",
        manifest_entry_hash=f"hash_{world_uid}",
    )


def _snapshot_hash(*objs) -> str:
    """Canonical cross-process exact hash (same encoding as the
    train-loop state hash — identical tensors hash identical)."""
    h = hashlib.sha256()
    for obj in objs:
        h.update(_canonical_bytes(obj))
    return h.hexdigest()


def _next_task_hash(sampler_state: dict) -> str:
    """Layer 2: the next-task decision axes of the sampler state —
    cursors + role RNG + global indices decide exactly which task the
    sampler emits next (SEED-RESUME continues them across processes)."""
    keys = (
        "cursors", "role_sampler_rng", "global_attempt_index",
        "global_episode_index", "global_completed_index",
    )
    return _snapshot_hash(
        {k: sampler_state[k] for k in keys if k in sampler_state})


def _capture(res, update: int, state_before: str, policy, critic,
             actor_opt, critic_opt, sampler) -> dict:
    """Nine-layer exact snapshot of one committed update (guide Step 6)."""
    sampler_state = sampler.state_dict()
    buffer = res["buffer"]
    metrics = res["metrics"]
    layers = {
        "1_load_complete_state": state_before,
        "2_next_task_world_episode": _next_task_hash(sampler_state),
        "3_rollout_noise_action_chain_logprob": _snapshot_hash(
            buffer.chains, buffer.logprobs, buffer.episode_ids),
        "4_reward_outcome_buffer_next_values": _snapshot_hash(
            buffer.rewards, buffer.next_values,
            getattr(buffer, "next_value_provenance", None),
            metrics.get("outcome_counts")),
        "5_values_advantages_returns": _snapshot_hash(
            buffer.values, metrics.get("advantages_hash"),
            metrics.get("returns_hash")),
        "6_randperm_minibatch": _snapshot_hash(
            metrics.get("randperm_hash"), metrics.get("update_batches")),
        "7_actor_critic_grads": _snapshot_hash(
            metrics.get("grads_hash"),
            metrics.get("actor_grad_norm"),
            metrics.get("critic_grad_norm")),
        "8_parameters_optimizer_moments": _snapshot_hash(
            policy.state_dict(), critic.state_dict(),
            actor_opt.state_dict(), critic_opt.state_dict()),
        "9_sampler_slots_counters_metrics": _snapshot_hash(
            sampler_state, res["active_slot_states"],
            res["env_steps"], res["completed_episodes"],
            res["voided_episodes"], metrics.get("actor_stepped"),
            metrics.get("actor_lr_after"), metrics.get("critic_lr_after"),
            metrics.get("actor_scheduler_step_count"),
            metrics.get("critic_scheduler_step_count"),
            metrics.get("reward_scaler_step_count"),
            res.get("metrics_writer_cursor")),
    }
    upd = int(re.search(r"_ppoupd_(\d+)\.pt$", res["checkpoint_path"]).group(1))
    return {
        "update": update,
        "layers": layers,
        "meta": {
            "ppo_update_count": upd,
            "checkpoint_present": bool(res["checkpoint_path"]),
            "checkpoint_path": res["checkpoint_path"],
        },
    }


def _real_world(device: str):
    """Production components for --real-gpu (Zread P1): frozen-loader BC
    policy + registered LoRA + E040 critic + real UAVVectorEnvAdapter
    over the admission pool, plus the production preflight facts
    (validate_official_environment — requires a clean tree, the evidence
    discipline).  Returns the tuple run_e2e_batch consumes plus the
    runtime dict built from the facts."""
    from diffusion_policy_3d.policy.lora_adapters import insert_lora
    from diffusion_policy_3d.rl.dppo_uav.critic import (
        PolicyObsEncoder,
        ValueCritic,
    )
    from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
        validate_official_environment,
    )
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        manifest_record_key,
    )
    from diffusion_policy_3d.rl.dppo_uav.integration import (
        build_adapter,
        build_task_specs,
        entry_metadata_hash,
        manifest_train_entries,
    )
    from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy

    bc = os.environ.get("DPPO_BC_CKPT", DEFAULT_BC_CKPT)
    manifest = os.environ.get("DPPO_MANIFEST", DEFAULT_MANIFEST)
    worlds = os.environ.get("DPPO_WORLDS_TRAIN", DEFAULT_WORLDS_TRAIN)
    facts = validate_official_environment(
        manifest_path=manifest, bc_checkpoint_path=bc,
        worlds_train_dir=worlds, repo_root=REPO_ROOT,
    )
    contract = RLSystemContract()
    policy = load_ddpm_policy(bc, device=device)
    lora_info = insert_lora(policy.model, r=8, alpha=8, verbose=False)
    torch.manual_seed(BASE_SEED)
    torch.cuda.manual_seed_all(BASE_SEED)
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
    lora_bundle = policy.model.dppo_lora_adapters
    actor_params = tuple(lora_bundle.trainable_parameters())
    assert actor_params, "registered LoRA bundle has no trainable parameters"
    actor_opt = torch.optim.AdamW(actor_params, lr=1e-4)
    critic_opt = torch.optim.AdamW(critic.value_head_parameters(), lr=1e-4)
    entries = manifest_train_entries(manifest)
    specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e))
         for e in entries])
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    adapter = build_adapter(
        contract,
        worlds_dir=worlds,
        manifest_path=manifest,
        manifest_records=records,
        device=device,
        max_episode_steps=60,
    )
    runtime = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "reward_version": "v16",
        "bc_checkpoint_hash": facts["bc_checkpoint_hash"],
        "manifest_hash": facts["manifest_hash"],
        "source_file_hashes": facts["source_file_hashes"],
        "project_git_commit": facts["project_git_commit"],
        "project_tree_hash": facts["project_tree_hash"],
        "world_assets_hash": facts["world_assets_hash"],
    }
    return (policy, lora_info, critic, actor_opt, critic_opt, adapter,
            specs, contract, runtime)


def _run_updates(out_dir: str, updates: int, phase: str,
                 resume_from: str = "", first_update: int = 1,
                 real_gpu: bool = False,
                 probe_out: str = "",
                 probe_capture_update: Optional[int] = None,
                 probe_expect_layer3: Optional[str] = None) -> int:
    """Run ``updates`` one_update batches, snapshotting every committed
    update to ``out_dir/update_{i}.json`` (absolute update numbers —
    resume_b starts at ``first_update``); the last update's checkpoint
    path goes to ``out_dir/manifest.json``.

    ``probe_*`` (F4 guide §3) attach the RootCauseProbe: it wraps
    train_loop.sample_dppo_rollout (the from-import bound name the loop
    actually calls) and the adapter's step_at, records per-update RNG +
    component hashes, clears at the requested capture update, and
    finalize() hard-fails when its layer-3 hash mismatches the expected
    reference (probe contract exit code folds into this function's)."""
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpts = outdir / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)
    # Step 6 same-seed protocol: every phase process seeds its GLOBAL
    # RNG identically BEFORE constructing anything (production seeding
    # semantics — the launcher seeds torch.manual_seed before workload).
    # Process-local determinism is exactly what the control side proves;
    # the resume side additionally restores checkpoint RNG (layer 9).
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    contract = RLSystemContract()
    configure_numerical_mode(FORMAL_NUMERICAL_MODE)
    if real_gpu:
        (policy, lora_info, critic, actor_opt, critic_opt, env, specs,
         contract, runtime) = _real_world("cuda")
        device = "cuda"
        n_episodes = 32
        max_ep_len = 60
    else:
        policy = FakePolicy(contract)
        critic = ValueCritic(
            contract=contract, encoder=PolicyObsEncoder(policy))
        actor_opt = torch.optim.AdamW(policy.model.parameters(), lr=1e-4)
        critic_opt = torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4)
        env = MockUAVEnv(contract)
        specs = [_spec(f"mock_{role}", role) for role in ROLE_ORDER]
        device = "cpu"
        n_episodes = 2
        max_ep_len = 12
        lora_info = {}
        runtime = {
            "timestamp": "2026-08-24T00:00:00Z",
            "device": "cpu",
            "reward_version": "v1.3",
            "bc_checkpoint_hash": "bc0123",
            "manifest_hash": "mf456",
            "source_file_hashes": {},
            "project_git_commit": "n/a",
            "project_tree_hash": "n/a",
            "world_assets_hash": "wa456",
        }
    cfg = LoopConfig(
        mode="one_update",
        n_episodes=n_episodes,
        max_ep_len=max_ep_len,
        device=device,
        output_dir=str(ckpts),
        eval_n_episodes=0,
    )
    sampler = EpisodeTaskSampler(
        specs, DEFAULT_ROLE_TARGETS, base_seed=BASE_SEED)
    kwargs = {}
    if resume_from:
        kwargs["resume_from"] = resume_from

    # PRODUCTION batch chaining (train_dppo_uav_official.py): consecutive
    # batches resume the previous batch's checkpoint — "idempotent no-op
    # simply continues the counters". update 1 is fresh on both sides;
    # resume_b's update 3 is the CROSS-PROCESS cold start.
    prev_ckpt = resume_from
    state_before = _state_hash(
        policy, critic, actor_opt, critic_opt, sampler)
    last_ckpt = ""
    probe = None
    layer3_by_update: dict = {}
    if probe_out:
        # F4 guide §3: the probe wraps the names the training loop ACTUALLY
        # resolves at call time — train_loop.sample_dppo_rollout (bound by
        # from-import at train_loop.py:108-109) and type(env).step_at.
        from diffusion_policy_3d.rl.dppo_uav import train_loop as _tl
        probe = RootCauseProbe(ProbeConfig(
            out_path=probe_out,
            capture_update=probe_capture_update,
            expect_layer3=probe_expect_layer3,
        ))
        probe.install_rollout_hook(_tl)
        probe.install_step_hook(type(env))
        probe.attach_env(env)
    for k in range(updates):
        i = first_update + k
        if probe is not None:
            probe.begin_update(i)
        kwargs = {"resume_from": prev_ckpt} if prev_ckpt else {}
        # R7-6 Step 9 cursor law: batch i saves cursor i-1 (records
        # committed before it) — the restore-side drift gate demands
        # cursor == outer_iteration-1.
        kwargs["metrics_writer_cursor"] = i - 1
        res = run_e2e_batch(
            env=env,
            policy=policy,
            critic=critic,
            contract=contract,
            loop_config=cfg,
            trainer_config=TrainerConfig(device=device),
            task_specs=specs,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
            lora_info=lora_info,
            runtime=runtime,
            task_sampler=sampler,
            **kwargs,
        )
        # Layer 1: the load-complete state — on any resumed batch this is
        # the state after restore_training_state completed (inside
        # run_e2e_batch); on the fresh update it is the construction
        # state. The comparator proves these coincide across processes.
        if prev_ckpt:
            state_before = res["restored_state_hash"]
        snapshot = _capture(
            res, i, state_before, policy, critic, actor_opt, critic_opt,
            sampler)
        layer3_by_update[i] = snapshot["layers"][
            "3_rollout_noise_action_chain_logprob"]
        (outdir / f"update_{i}.json").write_text(
            json.dumps(snapshot, indent=2))
        state_before = _state_hash(
            policy, critic, actor_opt, critic_opt, sampler)
        last_ckpt = res["checkpoint_path"]
        prev_ckpt = last_ckpt
    if probe is not None:
        # finalize against the layer-3 hash of the captured update — the
        # same quantity the nine-layer comparator records (buffer.chains/
        # logprobs/episode_ids), so the probe's own data validity is
        # cross-checked against the reference before the GPU gate.
        target_update = probe_capture_update if probe_capture_update else i
        layer3 = layer3_by_update.get(target_update)
        rc = probe.finalize(layer3=layer3)
        probe.uninstall()
        if rc != 0:
            print(f"[{phase}] probe contract FAILED (rc={rc}); "
                  f"see {probe_out}")
            return rc
    (outdir / "manifest.json").write_text(
        json.dumps({"ckpt_path": last_ckpt, "updates": i}))
    print(f"[{phase}] updates 1..{updates} snapshotted; "
          f"last ckpt {last_ckpt}")
    return 0


def _load_snapshots(workdir: Path, phase: str,
                    first: int, last: int) -> dict:
    out = {}
    for i in range(first, last + 1):
        path = workdir / phase / f"update_{i}.json"
        out[i] = json.loads(path.read_text())
    return out


def phase_compare(workdir: Path) -> int:
    control = _load_snapshots(workdir, "control", 1, UPDATES)
    resume = dict(_load_snapshots(workdir, "resume_a", 1, RESUME_SPLIT))
    resume.update(_load_snapshots(
        workdir, "resume_b", RESUME_SPLIT + 1, UPDATES))
    metric_diffs = 0
    ckpt_diffs = 0
    first_diff = None
    compared: list = []
    for i in range(1, UPDATES + 1):
        c, r = control[i], resume[i]
        for layer in LAYERS:
            compared.append(layer)
            if c["layers"][layer] != r["layers"][layer]:
                metric_diffs += 1
                first_diff = {
                    "update": i,
                    "layer": layer,
                    "control": c["layers"][layer],
                    "resume": r["layers"][layer],
                    "detail": "layer hash mismatch",
                }
                break
        if first_diff is not None:
            break
        # Step 7 exact gate — checkpoint presence + update numbers.
        for side in ("control", "resume"):
            meta = c["meta"] if side == "control" else r["meta"]
            if not meta["checkpoint_present"]:
                ckpt_diffs += 1
                first_diff = {
                    "update": i, "layer": "checkpoint_presence",
                    "control": str(c["meta"]["checkpoint_present"]),
                    "resume": str(r["meta"]["checkpoint_present"]),
                    "detail": f"{side} checkpoint missing at update {i}",
                }
                break
            if meta["ppo_update_count"] != i:
                ckpt_diffs += 1
                first_diff = {
                    "update": i, "layer": "checkpoint_update_number",
                    "control": str(c["meta"]["ppo_update_count"]),
                    "resume": str(r["meta"]["ppo_update_count"]),
                    "detail": f"{side} checkpoint update number {meta['ppo_update_count']} != {i}",
                }
                break
        if first_diff is not None:
            break
    pass_ok = (metric_diffs == 0 and ckpt_diffs == 0 and first_diff is None)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "pass": pass_ok,
        "metric_diffs": metric_diffs,
        "ckpt_diffs": ckpt_diffs,
        "first_diff": first_diff,
        "control": {
            "ckpt": control[UPDATES]["meta"]["checkpoint_path"],
            "updates": UPDATES,
        },
        "resume": {
            "ckpt": resume[UPDATES]["meta"]["checkpoint_path"],
            "updates": UPDATES,
        },
        "layers_compared": sorted(set(compared)),
    }
    (workdir / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if pass_ok else 1


def _phase_resume_b(workdir: Path, *, real_gpu: bool = False,
                    probe_out: str = "",
                    probe_capture_update: Optional[int] = None,
                    probe_expect_layer3: Optional[str] = None) -> int:
    manifest = json.loads(
        (workdir / "resume_a" / "manifest.json").read_text())
    resume_from = manifest["ckpt_path"]
    return _run_updates(
        str(workdir / "resume_b"), UPDATES - RESUME_SPLIT,
        "resume_b", resume_from=resume_from,
        first_update=RESUME_SPLIT + 1, real_gpu=real_gpu,
        probe_out=probe_out,
        probe_capture_update=probe_capture_update,
        probe_expect_layer3=probe_expect_layer3)


def phase_all(workdir: Path, *, real_gpu: bool = False) -> int:
    here = [sys.executable, "-u", str(Path(__file__).resolve())]
    if real_gpu:
        here.append("--real-gpu")
    for phase in ("control", "resume_a", "resume_b", "compare"):
        proc = subprocess.run(
            here + ["--phase", phase, "--workdir", str(workdir)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"[--all] {phase} failed:\n{proc.stdout}\n{proc.stderr}")
            return proc.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--phase",
        choices=("control", "resume_a", "resume_b", "compare"),
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--real-gpu",
        action="store_true",
        help="F4 (Zread P1): real BC policy + registered LoRA + "
        "UAVVectorEnvAdapter on CUDA (requires a clean tree — the "
        "production preflight validate_official_environment enforces it)",
    )
    parser.add_argument(
        "--probe-out", default="",
        help="F4 guide §3: write the RootCauseProbe artifact here; the "
        "probe wraps train_loop.sample_dppo_rollout + adapter.step_at, "
        "recording per-update RNG and split observation/action hashes. "
        "Empty disables the probe. On CPU the probe fails closed "
        "(MockUAVEnv lacks the real env internals) — use with --real-gpu.",
    )
    parser.add_argument(
        "--probe-capture-update", type=int, default=None,
        help="Absolute update number the probe clears at and records "
        "from (probe-9 concatenation defect); default: record everything.",
    )
    parser.add_argument(
        "--probe-expect-layer3", default=None,
        help="Expected layer-3 hash (rollout_noise_action_chain_logprob) "
        "for the captured update — finalize() hard-fails on mismatch "
        "(probe data validity gate).",
    )
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if args.all:
        return phase_all(workdir, real_gpu=args.real_gpu)
    probe_kw = {
        "probe_out": args.probe_out,
        "probe_capture_update": args.probe_capture_update,
        "probe_expect_layer3": args.probe_expect_layer3,
    }
    if args.phase == "control":
        return _run_updates(str(workdir / "control"), UPDATES, "control",
                            real_gpu=args.real_gpu, **probe_kw)
    if args.phase == "resume_a":
        return _run_updates(
            str(workdir / "resume_a"), RESUME_SPLIT, "resume_a",
            real_gpu=args.real_gpu, **probe_kw)
    if args.phase == "resume_b":
        return _phase_resume_b(workdir, real_gpu=args.real_gpu, **probe_kw)
    if args.phase == "compare":
        return phase_compare(workdir)
    parser.error("one of --phase / --all is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
