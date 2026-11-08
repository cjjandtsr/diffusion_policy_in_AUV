"""role_metrics_probe.py — production role-metrics evidence (R2 rework P1-2).

Audit finding (P2-10): "ROLE-TRANSITION 无 production metrics JSON" — the
R2 closure cited G3_GATE_REPORT_v2.json for the ROLE-TRANSITION row, but
that report carries RESET counts only; it never records per-role
attempted/completed/voided/transition. This probe persists the production
truthful per-role statistics as a standalone evidence JSON.

Production path (everything under evidence is REAL):
  EpisodeTaskSampler.next_attempt() -> UAVVectorEnvAdapter.reset/step
    -> PointCloudValidityGate (per frame) -> run_e2e_batch role_stats
    (train_loop.py:964-996, the same accounting the formal train loop
    produces).
The action generator is the CPU FakePolicy fixture (tests/dppo_parity,
the accepted real-env integration pattern: "the mock env is not
involved") — action generation is NOT under evidence; sampler, adapter,
gate, manifest, and accounting all are. Episode faults are structurally
impossible on this path (R2.5 rework: invalid=0 at 10k), so the voided
branch is exercised by the unit suite (test_role_stats_truthful_accounting)
and the no-ray stress; a run with voided>0 would still be recorded
truthfully.

Pass condition: the batch completes (no AdmissionExhausted /
EpisodeCollectionIncomplete) AND the accounting invariants hold:
  attempted >= completed + voided (multi-slot: attempted counts
  RESETS; episodes in-flight at the collection bound are dropped —
  recorded truthfully as in_flight_at_collection_stop, not masked);
  completed == n_episodes (full episode collection — voided>0 fails
  the probe on this production path, where voided is structurally 0);
  sum(transition_counts) == env_steps  (voided episodes enter no
  transitions — PC-VALID "invalid buffer transition=0");
  zero invalid point-cloud frames.
Evidence JSON binds command / seed / exit code / tree hash (P2-10) and
records the REAL adapter construction arguments: requested_n_envs vs
resolved_n_envs (= env.n_envs), verified equal before any rollout
(R2 closure rework #2, 审核裁决 P1).

Run (after `git add` so the tree hash is complete):
  python scripts/role_metrics_probe.py --episodes 120 --n-envs 4 \
      --max-ep-len 100 --seed 0 \
      --output program/reports/routeA_rl_env_v1/g3_gate/ROLE_METRICS_EVIDENCE.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# FakePolicy lives with the parity fixtures (same as continuation_probe.py).
_TESTS = os.path.join(REPO_ROOT, "tests", "dppo_parity")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionExhausted,
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_task_specs,
    entry_metadata_hash,
    manifest_train_entries,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    EpisodeCollectionIncomplete,
    LoopConfig,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy

# Production defaults (same convention as g3_10k_reset_gate_audit.py).
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)
PRODUCTION_EPISODES = 120
PRODUCTION_N_ENVS = 4
PRODUCTION_MAX_EP_LEN = 100  # E040 formal training default (train_ddpm_ppo_meanlogp_3_e040.py --max_ep_len 100)
ACTION_SCALE = 0.0002  # integration-test-probed scale: in-world at seed pins

# role targets
ROLE_NAMES = ("anticipatory", "dense_recovery", "support_expansion",
              "yaw_lateral_recovery")


def _runtime(device: str, seed: int) -> Dict[str, Any]:
    return {
        "timestamp": "2026-08-09T00:00:00Z",
        "device": device,
        "reward_version": "v16",
        "bc_checkpoint_hash": "n/a",
        "world_assets_hash": "n/a",
        "manifest_hash": "n/a",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
        "probe_seed": seed,
    }


def _tree_hash() -> str:
    """The exact git tree the evidence was produced from (P2-10 binding)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], capture_output=True,
            text=True, cwd=REPO_ROOT,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "not-a-git-repo"


def _records(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Composite-key -> source-entry hash, recomputed from the manifest
    ENTRY (mirrors gate_audit.source_hash_records: any divergence in
    build_task_specs surfaces as a hash-mismatch instead of self-copy)."""
    return {
        manifest_record_key(
            e["world_uid"],
            e["role"],
            e.get("admission_id", f"{e['world_uid']}__{e['role']}__{e['split']}"),
        ): entry_metadata_hash(e)
        for e in entries
    }


def run_probe(
    *,
    manifest: str,
    worlds_dir: str,
    episodes: int,
    n_envs: int,
    max_ep_len: int,
    seed: int,
) -> Tuple[Dict[str, Any], int]:
    """Run one no-update e2e batch over the production path and return the
    (evidence, exit_code) pair. exit 0 iff pass (see module docstring)."""
    # Deterministic: pin the global RNGs (FakePolicy draws the DDIM initial
    # noise from the global torch RNG — the real-env integration pattern).
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    contract = RLSystemContract()
    entries = manifest_train_entries(manifest)
    # g3 pattern: the production manifest carries no per-entry hash field
    # (ASSUMPTION-6 hash-proxy), so the recomputed source hash is injected
    # before spec construction — the adapter's rule-6 records use the same
    # recomputed hashes, so any divergence is a hash-mismatch, never a
    # masked self-match.
    train_specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries]
    )
    sampler = EpisodeTaskSampler(
        train_specs, DEFAULT_ROLE_TARGETS, base_seed=seed
    )
    env = build_adapter(
        contract, worlds_dir=worlds_dir, manifest_path=manifest,
        manifest_records=_records(entries),
        device="cpu", n_envs=n_envs, max_episode_steps=max_ep_len,
    )
    if int(env.n_envs) != n_envs:
        # R2 closure rework #2 (审核裁决 2026-08-09 P1): the evidence must
        # never claim slots the adapter did not actually construct — the
        # old probe ran a single-slot adapter (default n_envs=1) while
        # reporting n_envs=4.
        raise RuntimeError(
            f"n_envs wiring error: requested {n_envs} but resolved "
            f"env.n_envs={int(env.n_envs)}")
    # run_e2e_batch asserts the numerical mode is already configured
    # (single configuration source, guide R7-3 Step 1) — configure before
    # policy construction / any forward.
    from diffusion_policy_3d.rl.dppo_uav.numerics import (
        configure_numerical_mode,
    )
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        FORMAL_NUMERICAL_MODE,
    )

    configure_numerical_mode(FORMAL_NUMERICAL_MODE)
    policy = FakePolicy(contract, action_scale=ACTION_SCALE)
    critic = ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))

    evidence: Dict[str, Any] = {
        "title": "R2 rework P1-2: production role metrics "
                 "(attempted/completed/voided/transition)",
        "n_episodes": episodes,
        "requested_n_envs": n_envs,
        "resolved_n_envs": int(env.n_envs),  # real adapter construction arg
        "max_ep_len": max_ep_len, "seed": seed,
        "rollout_path": "EpisodeTaskSampler -> UAVVectorEnvAdapter "
                        "reset/step -> PointCloudValidityGate -> "
                        "run_e2e_batch role_stats (no_update); policy = "
                        "CPU FakePolicy action generator (not under evidence)",
        "command": " ".join(sys.argv),
        "tree_hash": _tree_hash(),
    }
    with tempfile.TemporaryDirectory(prefix="role_metrics_probe_") as tmp:
        try:
            loop_config = LoopConfig(
                mode="no_update", n_episodes=episodes, max_ep_len=max_ep_len,
                device="cpu", output_dir=tmp,
            )
            res = run_e2e_batch(
                env=env, policy=policy, critic=critic, contract=contract,
                loop_config=loop_config, trainer_config=TrainerConfig(
                    device="cpu"),
                task_specs=train_specs,
                actor_optimizer=torch.optim.AdamW(
                    policy.model.parameters(), lr=1e-4),
                critic_optimizer=torch.optim.AdamW(
                    critic.value_head_parameters(), lr=1e-4),
                lora_info={"lora_params": []},
                runtime=_runtime("cpu", seed),
                task_sampler=sampler,
            )
        except (AdmissionExhausted, EpisodeCollectionIncomplete) as exc:
            evidence.update({
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
                "exit_code": 1,
            })
            return evidence, 1

    rs = res["role_stats"]
    attempted_t = sum(rs["attempted_episode_counts"].values())
    completed_t = sum(rs["completed_episode_counts"].values())
    voided_t = sum(rs["voided_episode_counts"].values())
    transitions_t = sum(rs["transition_counts"].values())
    accounting = {
        "attempted_total": attempted_t,
        "completed_total": completed_t,
        "voided_total": voided_t,
        "rejected_total": res["rejected_episodes"],
        "transitions_total": transitions_t,
        "env_steps": res["env_steps"],
        "completed_episodes": res["completed_episodes"],
        "voided_episodes": res["voided_episodes"],
        # Rework #3 (审核裁决 P1-1): the precise refill budget keeps
        # completed + active <= n_episodes at every step, so at collection
        # stop attempted == completed + voided EXACTLY and nothing is left
        # in flight (the rework-#2-era tail drop — attempted 123 for 120
        # requested — no longer exists; in_flight is recorded truthfully
        # and must be zero).
        "in_flight_at_collection_stop":
            attempted_t - completed_t - voided_t,
    }
    invariants = {
        # Rework #3: exact accounting at the precise budget — no attempt
        # over-shoot, nothing dropped at collection stop.
        "attempted_eq_completed_plus_voided":
            attempted_t == completed_t + voided_t,
        "in_flight_zero":
            attempted_t - completed_t - voided_t == 0,
        "transitions_eq_env_steps":
            transitions_t == res["env_steps"],
        "invalid_frames_zero":
            res["gate_summary"]["invalid_count"] == 0,
        # the batch collects EVERY requested episode exactly: attempted ==
        # n_episodes (over-refill gone) and completed == n_episodes
        # (collection-completeness bar). voided is structurally 0 on this
        # production path (R2.5 invalid=0 at 10k) and asserted so — a run
        # with faulted episodes would fail the probe.
        "attempted_eq_n_episodes":
            attempted_t == episodes,
        "completed_eq_n_episodes":
            completed_t == episodes,
        "voided_zero":
            voided_t == 0,
    }
    ok = all(invariants.values())
    evidence.update({
        "pass": ok,
        "role_stats": rs,
        "accounting": accounting,
        "invariants": invariants,
        "gate_invalid_count": res["gate_summary"]["invalid_count"],
        "exit_code": 0 if ok else 1,
    })
    return evidence, 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None,
                        help="RouteA RL v1 admission manifest JSON "
                             "(default: $DPPO_MANIFEST or the production path)")
    parser.add_argument("--worlds-dir", default=None,
                        help="train worlds dir with <uid>.world.obstacles.json "
                             "(default: $DPPO_WORLDS_TRAIN or the production path)")
    parser.add_argument("--episodes", type=int, default=PRODUCTION_EPISODES)
    parser.add_argument("--n-envs", type=int, default=PRODUCTION_N_ENVS)
    parser.add_argument("--max-ep-len", type=int, default=PRODUCTION_MAX_EP_LEN)
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.episodes % args.n_envs != 0:
        # P2-8: a partial final batch must never be reported as success.
        print(f"n_episodes ({args.episodes}) must be a multiple of n_envs "
              f"({args.n_envs})", file=sys.stderr)
        return 1
    manifest = args.manifest or os.environ.get(
        "DPPO_MANIFEST", DEFAULT_MANIFEST)
    worlds_dir = args.worlds_dir or os.environ.get(
        "DPPO_WORLDS_TRAIN", DEFAULT_WORLDS_TRAIN)

    print(f"role-metrics probe: {args.episodes} episodes x "
          f"max_ep_len={args.max_ep_len}, {args.n_envs} envs, seed={args.seed}",
          flush=True)
    evidence, rc = run_probe(
        manifest=manifest, worlds_dir=worlds_dir, episodes=args.episodes,
        n_envs=args.n_envs, max_ep_len=args.max_ep_len, seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"\nROLE METRICS {'PASS' if rc == 0 else 'FAIL'} "
          f"(exit {rc}): {evidence.get('error', '')}")
    print(f"evidence: {args.output}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
