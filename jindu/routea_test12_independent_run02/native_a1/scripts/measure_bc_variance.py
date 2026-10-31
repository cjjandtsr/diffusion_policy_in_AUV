"""measure_bc_variance.py — the frozen-bank BC action-variance baseline
(review R5.6 / §8.6: the G7 action-variance gate must compare against a
BC-on-bank measurement via the E1-paired pipeline, NOT the first RL
batch).

Runs the frozen paired bank with BC on BOTH sides (identical policy,
paired policy-noise RNG per entry), collects every executed action, and
writes {action_variance_mean, per_dim_variance, n_episodes, n_actions,
bank_id, rng_contract_version}.

Run:
  python scripts/measure_bc_variance.py \
      --paired_bank <bank.json> --bc-ckpt <BC ckpt> \
      --out <bc_variance.json> --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract  # noqa: E402
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (  # noqa: E402
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (  # noqa: E402
    file_sha256,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (  # noqa: E402
    read_paired_bank_file,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (  # noqa: E402
    build_adapter,
    build_routea_eval_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_eval_entries,
)
from diffusion_policy_3d.rl.dppo_uav.paired_rng import (  # noqa: E402
    RNG_CONTRACT_VERSION,
    policy_noise_seed,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy  # noqa: E402
from diffusion_policy_3d.rl.dppo_uav.train_loop import (  # noqa: E402
    LoopConfig,
    run_paired_eval,
)

DEFAULT_BC_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    "/checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired_bank", required=True)
    parser.add_argument("--bc_ckpt", default=DEFAULT_BC_CKPT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--worlds", default=DEFAULT_WORLDS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_ep_len", type=int, default=60)
    parser.add_argument(
        "--rl_ckpt", default=None,
        help="optional RL checkpoint whose exact_resolved_config resolves "
        "the eval-consistent floors/max_ep_len (the canonical eval reads "
        "the same erc — without it the explicit args apply)",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    contract = RLSystemContract()
    bank = read_paired_bank_file(args.paired_bank)
    with open(args.paired_bank, encoding="utf-8") as f:
        bank_id = json.load(f).get("bank_id", "")

    entries = manifest_eval_entries(args.manifest)
    specs_eval, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e))
         for e in entries]
    )
    pool = build_routea_eval_pool(args.manifest, args.worlds)
    records_eval = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs_eval
    }
    eval_env = build_adapter(
        contract, worlds_dir=args.worlds, pool=pool,
        manifest_records=records_eval, device=args.device,
        max_episode_steps=args.max_ep_len, n_envs=1,
    )
    bc_policy = load_ddpm_policy(args.bc_ckpt, device=args.device)
    # R5.8-E (E-3): floors and max_ep_len must come from the SAME source
    # the canonical eval uses — the RL checkpoint's exact_resolved_config
    # — so the BC variance baseline is measured under the identical
    # sampling/episode conditions as the paired eval's BC runs.
    max_ep_len = args.max_ep_len
    min_sampling_std = 0.1
    min_logprob_std = 0.1
    if args.rl_ckpt:
        from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
            CheckpointBuilder,
        )

        erc = CheckpointBuilder.load(args.rl_ckpt).exact_resolved_config
        max_ep_len = int(erc["max_ep_len"])
        min_sampling_std = float(erc["min_sampling_denoising_std"])
        min_logprob_std = float(erc["min_logprob_denoising_std"])
    loop_config = LoopConfig(
        mode="one_update", n_episodes=1, max_ep_len=max_ep_len,
        device=args.device, output_dir="/tmp/bc_variance",
        eval_n_episodes=len(bank),
        min_sampling_denoising_std=min_sampling_std,
        min_logprob_denoising_std=min_logprob_std,
    )
    # R6-D §11.5.1 (R6-EVAL-1): both sides ARE the BC policy — the record
    # carries the BC checkpoint hash so dedup sees one policy identity
    # (the same BC hash the doc embeds at line 165).
    records, invalid = run_paired_eval(
        eval_env, bc_policy, bc_policy, contract, loop_config, bank,
        bank_id=bank_id,
        policy_checkpoint_hash=file_sha256(Path(args.bc_ckpt)),
    )
    if invalid:
        print(f"bc variance measurement: {invalid} invalid episodes",
              file=sys.stderr)
        return 1

    # Collect the executed actions from the paired trace: the record's
    # trace steps carry {action_norm, action_max_abs} summaries only, so
    # the per-dim variance comes from re-running the episodes with the
    # same derived seeds and recording the raw actions directly.
    action_rows = []
    for entry in bank:
        pn_seed = policy_noise_seed(
            bank_id, entry.episode_config_hash, replica_id=0
        )
        action_rows.append(_collect_entry_actions(
            eval_env, bc_policy, contract, entry, pn_seed, args
        ))
    actions = np.concatenate(action_rows, axis=0)  # (N, 11)
    per_dim = np.var(actions, axis=0)
    doc = {
        "action_variance_mean": float(np.mean(per_dim)),
        "per_dim_variance": [float(v) for v in per_dim],
        "n_actions": int(actions.shape[0]),
        "n_episodes": len(bank) - invalid,
        "bank_id": bank_id,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        # R5.8-E (E-3): CONTENT hash — a path string under a _hash name
        # cannot attest which BC checkpoint produced these numbers.
        "bc_checkpoint_hash": file_sha256(Path(args.bc_ckpt)),
        "world_assets_hash": "n/a",
        "action_space": "physical",
        "max_ep_len": int(max_ep_len),
        "min_sampling_denoising_std": float(min_sampling_std),
        "min_logprob_denoising_std": float(min_logprob_std),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))
    return 0


def _collect_entry_actions(eval_env, policy, contract, entry, pn_seed,
                           args) -> np.ndarray:
    """Run one bank entry with the derived policy-noise seed and collect
    the executed (unormalized) actions per step."""
    import torch

    from diffusion_policy_3d.rl.dppo_uav.paired_rng import paired_rng_context
    from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
        sample_dppo_rollout,
    )

    spec = entry.to_spec()
    rows = []
    with paired_rng_context(pn_seed):
        obs, info = eval_env.reset([spec], [entry.seed])
        steps = 0
        while True:
            from diffusion_policy_3d.rl.dppo_uav.train_loop import _selected

            # _selected: numpy -> tensor conversion + latest-window
            # selection (the same helper the paired-eval loop uses).
            sel, _ = _selected(obs, args.device, contract.obs_steps)
            with torch.no_grad():
                rollout = sample_dppo_rollout(
                    policy, sel, contract,
                    min_sampling_denoising_std=0.1,
                    min_logprob_denoising_std=0.1,
                )
            executed = policy.normalizer["action"].unnormalize(
                rollout.executed_action[0]
            )
            if torch.is_tensor(executed):
                executed = executed.detach().cpu().numpy()
            executed = np.asarray(executed, dtype=np.float32)
            rows.append(executed.reshape(-1, contract.action_dim))
            obs, reward, terminated, truncated, _ = eval_env.step(
                executed[None]
            )
            steps += 1
            if bool(np.asarray(terminated)[0]) or bool(
                np.asarray(truncated)[0]
            ) or steps >= args.max_ep_len:
                break
        if eval_env.is_active(0):
            eval_env.abort_episode([0])
    return np.concatenate(rows, axis=0)


if __name__ == "__main__":
    raise SystemExit(main())
