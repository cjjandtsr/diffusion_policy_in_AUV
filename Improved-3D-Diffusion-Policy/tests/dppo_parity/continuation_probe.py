"""continuation_probe.py — two-process SEED-RESUME continuation probe (R2 P1-3).

CPU probe: runs one run_e2e_batch (mode="one_update") in THIS process and
writes the consumed episode seeds (parsed from the rollout buffer's
episode ids, which P2-9 defines as ``world_uid:role:episode_seed``) plus
the checkpoint path as JSON. The pytest driver
(test_continuation_probe.py) runs the probe TWICE — the second time with
--resume-from pointing at the first run's checkpoint — and asserts the
second PROCESS's seeds strictly continue the first's:

    max(run1_seeds) < min(run2_seeds)

The production resume contract (SEED-RESUME): an interrupted run restarts
from its checkpoint with the sampler's attempt/cursor/RNG state restored,
so episode seeds are NEVER reused across a process boundary. Before the
P1-3 rework, resume_from restored policy/critic/optimizers/counters but
NOT the sampler — the fresh sampler restarted at base_seed and the resumed
run re-sampled the same episode seeds.

The probe mirrors the g6_one_update_resume_probe two-process pattern on
CPU (no CUDA, no real BC checkpoint), so the contract is enforced in the
CPU regression suite.

Run:
  python tests/dppo_parity/continuation_probe.py --out <json> \
      [--resume-from <ckpt.pt>] [--base-seed 1000] [--n-episodes 2]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    FORMAL_NUMERICAL_MODE,
    LoopConfig,
    configure_numerical_mode,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv

ROLE_ORDER = (
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
)


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


def run_probe(
    out_path: str,
    resume_from: str = "",
    base_seed: int = 1000,
    n_episodes: int = 2,
) -> int:
    contract = RLSystemContract()
    # Formal numerical mode must be configured before run_e2e_batch
    # (single configuration source, guide R7-3 Step 1 / R7-6 Step 3).
    configure_numerical_mode(FORMAL_NUMERICAL_MODE)
    policy = FakePolicy(contract)
    critic = ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))
    actor_opt = torch.optim.AdamW(policy.model.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(critic.value_head_parameters(), lr=1e-4)

    outdir = Path(out_path).resolve().parent
    cfg = LoopConfig(
        mode="one_update",
        n_episodes=n_episodes,
        max_ep_len=12,
        device="cpu",
        output_dir=str(outdir / "ckpts"),
        eval_n_episodes=0,
    )
    specs = [_spec(f"mock_{role}", role) for role in ROLE_ORDER]
    sampler = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=base_seed)
    runtime = {
        "timestamp": "2026-08-09T00:00:00Z",
        "device": "cpu",
        "reward_version": "v1.3",
        "bc_checkpoint_hash": "bc0123",
        "manifest_hash": "mf456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
        "world_assets_hash": "wa456",
    }
    kwargs = {}
    if resume_from:
        kwargs["resume_from"] = resume_from
    res = run_e2e_batch(
        env=MockUAVEnv(contract),
        policy=policy,
        critic=critic,
        contract=contract,
        loop_config=cfg,
        trainer_config=TrainerConfig(device="cpu"),
        task_specs=specs,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        lora_info={},
        runtime=runtime,
        task_sampler=sampler,
        **kwargs,
    )
    # P2-9: buffer episode ids are world_uid:role:episode_seed — the last
    # segment is the sampler's episode seed (base_seed + attempt).
    seeds = sorted(
        int(ep.split(":")[-1]) for ep in set(res["buffer"].episode_ids)
    )
    payload = {
        "episode_seeds": seeds,
        "checkpoint_path": res["checkpoint_path"],
        "env_steps": res["env_steps"],
        "completed_episodes": res["completed_episodes"],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--n-episodes", type=int, default=2)
    args = parser.parse_args()
    return run_probe(
        args.out, resume_from=args.resume_from,
        base_seed=args.base_seed, n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    sys.exit(main())
