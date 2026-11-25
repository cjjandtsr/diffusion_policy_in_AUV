"""g6_one_update_resume_probe.py — fresh-process resume smoke (G6, §5.8).

The G6 gate test proves resume parity in-process (continuous vs restored
world). This probe demonstrates the real usage pattern across PROCESS
boundaries, as an interrupted T6 micro-train would:

  stage1: one_update on a fresh world; writes checkpoint #1 (update 1).
  stage2: FRESH process (separate invocation): restore checkpoint #1 into
          a new world and run update 2 (resume_from=ckpt1), then continue
          in-process with update 3 (resume_from=ckpt2) — chained
          continuation, the exact micro-train loop.
  compare: load the three metrics JSONs: counters accumulate 1 -> 2 -> 3,
          all numeric metrics finite, per-batch env-steps add up.

Exit code 0 only when every check passes. Evidence (metrics JSON + the
checkpoints) stays in --outdir (default: temp/g6_probe) — probe evidence is
preserved, not cleaned up (P1 lesson from probe_e031_diag3).

Run:
  conda activate idp3
  python scripts/g6_one_update_resume_probe.py --mode stage1 --outdir temp/g6_probe
  python scripts/g6_one_update_resume_probe.py --mode stage2 --ckpt temp/g6_probe/<stage1-ckpt>.pt --outdir temp/g6_probe
  python scripts/g6_one_update_resume_probe.py --mode compare --outdir temp/g6_probe
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

DEFAULT_BC_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    "/checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"

PROBE_EPISODES = 16  # smaller than the gate's 32 — smoke, not gate
MAX_EP_LEN = 60
SEED = 0
MINIBATCH = 128  # G6 finding: 512 OOMs the 8 GiB card (pointnet recompute)


def log(msg: str) -> None:
    print(msg, flush=True)


def build_adapter() -> Tuple[Any, List[Any], RLSystemContract]:
    """Real adapter over the production admission pool (mirrors conftest)."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        manifest_record_key,
    )
    from diffusion_policy_3d.rl.dppo_uav.integration import (
        build_adapter as build,
        build_task_specs,
        entry_metadata_hash,
        manifest_train_entries,
    )

    entries = manifest_train_entries(DEFAULT_MANIFEST)
    assert entries, "no admitted rl_train entries in the manifest"
    specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries]
    )
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    contract = RLSystemContract()
    adapter = build(
        contract,
        worlds_dir=DEFAULT_WORLDS_TRAIN,
        manifest_path=DEFAULT_MANIFEST,
        manifest_records=records,
        device="cuda:0",
        max_episode_steps=MAX_EP_LEN,
    )
    return adapter, specs, contract


def fresh_world(device: str) -> Tuple[Any, Dict[str, Any], Any,
                                      torch.optim.Optimizer,
                                      torch.optim.Optimizer]:
    """Fresh BC policy + zero-init LoRA + E040 critic + optimizers."""
    from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    policy = load_ddpm_policy(DEFAULT_BC_CKPT, device=device)
    lora_info = insert_lora(policy.model, r=8, alpha=8, verbose=False)
    contract = RLSystemContract()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
    actor_optimizer = torch.optim.Adam(lora_info["lora_params"], lr=1e-3)
    critic_optimizer = torch.optim.Adam(
        critic.value_head_parameters(), lr=1e-3
    )
    return policy, lora_info, critic, actor_optimizer, critic_optimizer


def runtime(device: str) -> Dict[str, Any]:
    return {
        "device": device,
        "reward_version": "v16",
        "manifest_hash": "g6-probe",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "world_assets_hash": "?",
        "timestamp": "g6-probe",
    }


def metric_groups(metrics: Dict[str, Any]) -> List[str]:
    """Numeric metric values across the (possibly nested) metrics dict.

    bools are excluded: they are ints to isinstance and have no finite
    semantics (e.g. trainer early_stopped).
    """
    out: List[str] = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    out.append(f"{k}.{k2}={v2}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(f"{k}={v}")
    return out


def assert_finite(metrics: Dict[str, Any], tag: str) -> None:
    bad = [
        x for x in metric_groups(metrics)
        if not math.isfinite(float(x.split("=", 1)[1]))
    ]
    assert not bad, f"{tag}: non-finite metrics: {bad}"


def one_update(
    adapter: Any,
    specs: List[Any],
    contract: RLSystemContract,
    outdir: Path,
    tag: str,
    resume_from: str = "",
) -> Dict[str, Any]:
    """One one_update batch; returns (metrics, ckpt_path, env_steps)."""
    # run_e2e_batch asserts the numerical mode is already configured
    # (single configuration source, guide R7-3 Step 1) — configure before
    # policy load / LoRA insertion / optimizer construction.
    from diffusion_policy_3d.rl.dppo_uav.numerics import (
        configure_numerical_mode,
    )
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        FORMAL_NUMERICAL_MODE,
    )

    configure_numerical_mode(FORMAL_NUMERICAL_MODE)
    policy, lora_info, critic, opt_a, opt_c = fresh_world("cuda:0")
    loop_config = LoopConfig(
        mode="one_update", n_episodes=PROBE_EPISODES, max_ep_len=MAX_EP_LEN,
        device="cuda:0", output_dir=str(outdir / tag),
    )
    trainer_config = TrainerConfig(batch_size=MINIBATCH, device="cuda:0")
    kwargs = {}
    if resume_from:
        kwargs["resume_from"] = resume_from
    res = run_e2e_batch(
        adapter, policy, critic, contract, loop_config, trainer_config,
        specs, opt_a, opt_c, lora_info, runtime("cuda:0"), **kwargs,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=SEED
        ),
    )
    assert res["checkpoint_path"] is not None, f"{tag}: no checkpoint written"
    log(f"{tag}: env_steps={res['env_steps']} "
        f"completed={res['completed_episodes']} "
        f"rejected={res['rejected_episodes']} "
        f"routing={res['routing']}")
    log(f"{tag}: actor_grad_norm={res['metrics']['actor_grad_norm']:.6f} "
        f"critic_grad_norm={res['metrics']['critic_grad_norm']:.6f} "
        f"approx_kl={res['metrics']['approx_kl']:.6f}")
    return {
        "metrics": res["metrics"],
        "checkpoint_path": res["checkpoint_path"],
        "env_steps": res["env_steps"],
        "completed": res["completed_episodes"],
    }


def stage1(outdir: Path) -> None:
    adapter, specs, contract = build_adapter()
    res = one_update(adapter, specs, contract, outdir, "stage1")
    assert_finite(res["metrics"], "stage1")
    ckpt = Path(res["checkpoint_path"])
    log(f"stage1: ckpt={ckpt.name}")
    (outdir / "stage1_ckpt_path.txt").write_text(str(ckpt))
    (outdir / "stage1_metrics.json").write_text(json.dumps(
        res["metrics"], indent=2, default=str
    ))


def stage2(outdir: Path) -> None:
    ckpt1_txt = outdir / "stage1_ckpt_path.txt"
    assert ckpt1_txt.is_file(), "run --mode stage1 first"
    ckpt1 = ckpt1_txt.read_text().strip()
    adapter, specs, contract = build_adapter()

    # update 2 in a FRESH process world (restore from ckpt1)
    res2 = one_update(
        adapter, specs, contract, outdir, "stage2_resume", resume_from=ckpt1
    )
    assert_finite(res2["metrics"], "stage2_resume")
    log(f"stage2_resume: ckpt={Path(res2['checkpoint_path']).name}")

    # update 3: in-process continuation (chained micro-train usage)
    res3 = one_update(
        adapter, specs, contract, outdir, "stage2_continue",
        resume_from=res2["checkpoint_path"],
    )
    assert_finite(res3["metrics"], "stage2_continue")
    log(f"stage2_continue: ckpt={Path(res3['checkpoint_path']).name}")

    (outdir / "stage2_resume_metrics.json").write_text(json.dumps(
        res2["metrics"], indent=2, default=str
    ))
    (outdir / "stage2_continue_metrics.json").write_text(json.dumps(
        res3["metrics"], indent=2, default=str
    ))
    (outdir / "stage2_summary.json").write_text(json.dumps({
        "ckpt1": ckpt1,
        "ckpt2": res2["checkpoint_path"],
        "ckpt3": res3["checkpoint_path"],
        "env_steps": [res2["env_steps"], res3["env_steps"]],
        "completed": [res2["completed"], res3["completed"]],
    }, indent=2))


def compare(outdir: Path) -> None:
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    s1 = json.loads((outdir / "stage1_metrics.json").read_text())
    s2 = json.loads((outdir / "stage2_resume_metrics.json").read_text())
    s3 = json.loads((outdir / "stage2_continue_metrics.json").read_text())
    summary = json.loads((outdir / "stage2_summary.json").read_text())

    counts = [
        CheckpointBuilder.load(p).ppo_update_count
        for p in [summary["ckpt1"], summary["ckpt2"], summary["ckpt3"]]
    ]
    assert counts == [1, 2, 3], f"counter accumulation broken: {counts}"
    log(f"compare: ppo_update_count accumulation 1 -> 2 -> 3 OK")

    env_steps = [
        CheckpointBuilder.load(p).environment_transition_count
        for p in [summary["ckpt1"], summary["ckpt2"], summary["ckpt3"]]
    ]
    assert env_steps == sorted(env_steps) and len(set(env_steps)) == 3, (
        f"environment_transition_count must strictly accumulate: {env_steps}"
    )
    log(f"compare: env-step accumulation {env_steps} OK")

    # R2 P1-3: SEED-RESUME across process boundaries — the sampler's
    # attempt index must strictly accumulate through the checkpoint chain
    # (stage2_resume is a FRESH process resuming from ckpt1; a sampler that
    # restarted at base_seed would repeat attempt indices and re-issue the
    # same episode seeds).
    attempts = [
        CheckpointBuilder.load(p).sampler_state["global_attempt_index"]
        for p in [summary["ckpt1"], summary["ckpt2"], summary["ckpt3"]]
    ]
    assert attempts == sorted(attempts) and len(set(attempts)) == 3, (
        f"sampler global_attempt_index must strictly accumulate: {attempts}"
    )
    log(f"compare: sampler attempt continuation {attempts} OK")

    for tag, m in (("stage1", s1), ("stage2_resume", s2),
                   ("stage2_continue", s3)):
        assert_finite(m, tag)
        assert m["actor_grad_norm"] > 0.0, f"{tag}: actor grad 0"
        assert m["critic_grad_norm"] > 0.0, f"{tag}: critic grad 0"
        assert m["parameter_update_norm"] > 0.0, f"{tag}: no param update"
    log("compare: all metrics finite, all gradients > 0")
    log("G6 probe PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["stage1", "stage2", "compare"])
    parser.add_argument("--outdir", default=str(REPO_ROOT / "temp" / "g6_probe"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "probe requires CUDA"
    log(f"[g6 probe] mode={args.mode} outdir={outdir}")
    if args.mode == "stage1":
        stage1(outdir)
    elif args.mode == "stage2":
        stage2(outdir)
    else:
        compare(outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
