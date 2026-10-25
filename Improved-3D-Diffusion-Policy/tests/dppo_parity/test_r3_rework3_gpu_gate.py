"""RULING-7 (external-review #3) item ④: real-asset GPU integration gate.

Two gates, both on the REAL BC checkpoint + production admission pool
and a real CUDA device (skipped, not failed, when either is absent):

Gate 1 — fresh-process resume chain (Guide Step 4/5 counters + the
resume that RULING-6 claimed without evidence):
  - parent world A (continuous): update 1 -> save ckpt1 -> update 2;
  - a NEW python PROCESS B (fixed PYTHONHASHSEED, torch RNG restored
    from the checkpoint) restores ckpt1 and runs update 2;
  - update-2 metrics / advantages+returns hashes / counters /
    optimizer-step / completed-episode / sampler cursors must agree
    between A-continuous and B-fresh-process;
  - the schema-v4 checkpoint facts are pinned on the real save:
    n_lora_param_tensors == 2 * n_adaptors (real insert_lora: A+B per
    adaptor), lora_checksum binds the saved tensors, restorable=True
    with the fixed python_hash_seed recorded, sampler_state present.

Gate 2 — canonical paired eval on the REAL trained LoRA policy:
  - train one update with real identity facts in the runtime (the
    eval CLI refuses placeholder hashes);
  - freeze a paired bank on a dedicated n_envs=1 eval-pool adapter
    (freeze_paired_bank probes the real worlds), plus an offline
    density file;
  - run the official eval entry
    (scripts/eval_dppo_uav_official.py) as a SUBPROCESS — preflight
    (git clean / vendor hashes / numerical mode / PYTHONHASHSEED),
    checkpoint identity, LoRA wiring all exercised for real;
  - the persisted artifact must bind: checkpoint_hash, lora_checksum
    == live ckpt tensors, bank_id, offline_density_sha256, frozen
    episode_config_hash/manifest_entry_hash per record, raw per-step
    traces with the full TRACE_STEP_REQUIRED_KEYS schema, and
    use_obstacle_projection False.

These are evidence gates over already-committed rework #3 production
code (schema v4 + stable blake2b env RNG + hashseed preflight); a FAIL
is a production gap repaired test-first, a PASS is the RULING-7 item ④
evidence.

Memory discipline: the 8 GiB RTX 4060 card runs ONE training world at a
time — every phase releases its tensors (del + gc + empty_cache)
before the next phase (the child process and the eval CLI are separate
processes that must not coexist with the parent's policy on the card).

Run: pytest -m gpu tests/dppo_parity/test_r3_rework3_gpu_gate.py
"""

import gc
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import OfflineDensity
from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    EMPTY_LORA_CHECKSUM,
    CheckpointBuilder,
    contract_hash_of,
    lora_checksum_of,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
    file_sha256,
    official_source_file_hashes,
    world_assets_hash_of,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    TRACE_STEP_REQUIRED_KEYS,
    freeze_paired_bank,
    paired_bank_to_dict,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_eval_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_eval_entries,
    manifest_train_entries,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    OFFICIAL_DPPO_COMMIT,
    LoopConfig,
    restore_training_state,
    resume_identity_for,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BC_CKPT = str(
    REPO_ROOT
    / "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    / "checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"

EPISODES = 32          # G5/G6 gate scale
MAX_EP_LEN = 60
SEED = 0               # deterministic construction across worlds/processes
SAMPLER_BASE_SEED = 1  # sampler sequence replay (G6 gate-9 convention)
RANK = 8
ALPHA = 8
LR = 1e-3
BATCH = 128            # proven memory-safe on the 8 GiB card (G6 finding 1)

# resume parity tolerance (same op sequence on the same GPU; keep the
# small slack G6 uses for kernel scheduling differences)
RESUME_RTOL = 1e-4
RESUME_ATOL = 1e-6

DEVICE = "cuda:0"
SUBPROCESS_TIMEOUT = 3600

# capability roles the paired artifact computes metrics for — the bank
# must cover all three (capability_metrics hard-fails on a role with no
# record)
CAPABILITY_ROLES = ("anticipatory", "dense_recovery", "support_expansion")


@pytest.fixture(autouse=True)
def _gpu_cleanup():
    """Teardown frees the card even when a gate fails mid-test (RED) —
    the next gate in this module must never inherit a half-held world."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


def _assets() -> Tuple[str, str, str, str, str]:
    """Resolve the real assets; skip the gate when any is unavailable.

    R7-8 ruling 2026-08-24: the eval pool is the three literal splits
    rl_train ∪ rl_val ∪ rl_test — val/test world dirs are required
    assets for the eval-side tests (the bank must be freezeable over
    all three splits).
    """
    manifest = os.environ.get("DPPO_MANIFEST", DEFAULT_MANIFEST)
    worlds = os.environ.get("DPPO_WORLDS_TRAIN", DEFAULT_WORLDS_TRAIN)
    worlds_val = os.environ.get("DPPO_WORLDS_VAL", DEFAULT_WORLDS_VAL)
    worlds_test = os.environ.get("DPPO_WORLDS_TEST", DEFAULT_WORLDS_TEST)
    bc_ckpt = os.environ.get("DPPO_BC_CKPT", DEFAULT_BC_CKPT)
    if not torch.cuda.is_available():
        pytest.skip("gpu marker: CUDA unavailable")
    if not os.path.isfile(bc_ckpt):
        pytest.skip(f"BC checkpoint not found: {bc_ckpt}")
    if not os.path.isfile(manifest):
        pytest.skip(f"admission manifest not found: {manifest}")
    if not os.path.isdir(worlds):
        pytest.skip(f"worlds dir not found: {worlds}")
    if not os.path.isdir(worlds_val):
        pytest.skip(f"worlds val dir not found: {worlds_val}")
    if not os.path.isdir(worlds_test):
        pytest.skip(f"worlds test dir not found: {worlds_test}")
    return manifest, worlds, worlds_val, worlds_test, bc_ckpt


def _jsonable(x: Any) -> Any:
    """Canonical JSON-able form — both sides of a comparison run through
    the same converter, so equality is well-defined across the JSON
    boundary (np/torch types, tuples, sets, bytes)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, bytes):
        return x.hex()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def _fresh_world(device: str, bc_ckpt: str) -> Tuple[Any, Dict[str, Any],
                                                     Any, torch.optim.Optimizer,
                                                     torch.optim.Optimizer]:
    """Fresh BC policy + zero-init LoRA + E040 critic + optimizers.

    Deterministic construction: same seed sequence in every world /
    process, so construction never consumes RNG differently.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    policy = load_ddpm_policy(bc_ckpt, device=device)
    lora_info = insert_lora(policy.model, r=RANK, alpha=ALPHA, verbose=False)
    contract = RLSystemContract()
    torch.manual_seed(SEED)  # critic xavier init deterministic across worlds
    torch.cuda.manual_seed_all(SEED)
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
    actor_optimizer = torch.optim.Adam(lora_info["lora_params"], lr=LR)
    critic_optimizer = torch.optim.Adam(
        critic.value_head_parameters(), lr=LR
    )
    return policy, lora_info, critic, actor_optimizer, critic_optimizer


def _train_adapter(manifest: str, worlds: str) -> Tuple[Any, List[Any],
                                                        RLSystemContract]:
    """Real multi-slot training adapter over the admitted rl_train pool
    (the same construction the real_adapter fixture uses)."""
    entries = manifest_train_entries(manifest)
    specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries]
    )
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    contract = RLSystemContract()
    adapter = build_adapter(
        contract,
        worlds_dir=worlds,
        manifest_path=manifest,
        manifest_records=records,
        device=DEVICE,
        max_episode_steps=MAX_EP_LEN,
    )
    return adapter, specs, contract


def _release(*objs: Any) -> None:
    """Free GPU tensors before the next phase (one world per card)."""
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
# Gate 1: fresh-process resume chain
# ═══════════════════════════════════════════════════════════════════

# The child is a standalone script (no pytest) — it builds the same
# world, restores ckpt1 in a fresh process, runs update 2 and dumps the
# evidence JSON. PYTHONHASHSEED is pinned by the caller's env.
_CHILD_SCRIPT = """
import json
import os
import random
import sys

import numpy as np
import torch


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, bytes):
        return x.hex()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def main():
    repo_root, ckpt1, manifest, worlds, bc_ckpt, out_dir, out_json = sys.argv[1:8]
    sys.path.insert(0, repo_root)
    device = "cuda:0"

    from diffusion_policy_3d.policy.lora_adapters import insert_lora
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
        CheckpointBuilder, lora_checksum_of,
    )
    from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
    from diffusion_policy_3d.rl.dppo_uav.critic import (
        PolicyObsEncoder, ValueCritic,
    )
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        DEFAULT_ROLE_TARGETS, EpisodeTaskSampler, manifest_record_key,
    )
    from diffusion_policy_3d.rl.dppo_uav.integration import (
        build_adapter, build_task_specs, entry_metadata_hash,
        manifest_train_entries,
    )
    from diffusion_policy_3d.rl.dppo_uav.numerics import configure_numerical_mode
    from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        FORMAL_NUMERICAL_MODE, LoopConfig, restore_training_state,
        resume_identity_for, run_e2e_batch,
    )
    from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)
    policy = load_ddpm_policy(bc_ckpt, device=device)
    lora_info = insert_lora(policy.model, r=8, alpha=8, verbose=False)
    contract = RLSystemContract()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
    opt_a = torch.optim.Adam(lora_info["lora_params"], lr=1e-3)
    opt_c = torch.optim.Adam(critic.value_head_parameters(), lr=1e-3)

    entries = manifest_train_entries(manifest)
    specs, _ = build_task_specs([
        dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries
    ])
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    adapter = build_adapter(
        contract, worlds_dir=worlds, manifest_path=manifest,
        manifest_records=records, device=device,
        max_episode_steps=60,
    )
    loop_config = LoopConfig(
        mode="one_update", n_episodes=32, max_ep_len=60, device=device,
        output_dir=out_dir,
    )
    trainer_config = TrainerConfig(batch_size=128, device=device)
    # Identical placeholder runtime identity — the save side recorded the
    # same constants; the resume identity gate compares like for like.
    runtime = {
        "device": device,
        "reward_version": "v16",
        "manifest_hash": "r3-rework3-gpu-gate",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "world_assets_hash": "?",
        "timestamp": "r3-rework3-gpu-gate",
    }
    sampler = EpisodeTaskSampler(
        specs, DEFAULT_ROLE_TARGETS, base_seed=1
    )
    # Explicit cross-process restore: the checkpoint alone must rebuild
    # counters/sampler/RNG in a process that never saw the training run.
    restored = restore_training_state(
        ckpt1, policy, critic, lora_info, opt_a, opt_c,
        runtime_identity=resume_identity_for(
            contract, loop_config, trainer_config, runtime,
            configure_numerical_mode(FORMAL_NUMERICAL_MODE), policy,
            base_seed=1,
            n_envs=int(adapter.n_envs),
            lora_info=lora_info,
            actor_optimizer=opt_a,
            critic_optimizer=opt_c,
        ),
        task_sampler=sampler,
    )
    assert restored["ppo_update_count"] == 1, restored
    assert restored["environment_transition_count"] > 0, restored

    result = run_e2e_batch(
        adapter, policy, critic, contract, loop_config, trainer_config,
        specs, opt_a, opt_c, lora_info, runtime, resume_from=ckpt1,
        task_sampler=sampler,
    )
    ckpt2 = result["checkpoint_path"]
    assert ckpt2 is not None, "resumed update must write a checkpoint"
    saved2 = CheckpointBuilder.load(ckpt2)
    assert saved2.ppo_update_count == 2, saved2.ppo_update_count
    assert saved2.outer_iteration == 2, saved2.outer_iteration
    out = {
        "metrics": _jsonable(result["metrics"]),
        "env_steps": int(result["env_steps"]),
        "sampler_state": _jsonable(sampler.state_dict()),
        "ckpt2": {
            "ppo_update_count": saved2.ppo_update_count,
            "outer_iteration": saved2.outer_iteration,
            "optimizer_step_count": saved2.optimizer_step_count,
            "environment_transition_count":
                saved2.environment_transition_count,
            "completed_episode_count": saved2.completed_episode_count,
            "lora_checksum": lora_checksum_of(saved2.lora_state),
        },
    }

    # ── SECOND resume: the checkpoint chain must survive a second
    # fresh restore (Guide R3.2 Step 4: fresh→update1→resume update2→
    # resume update3, outer_iteration=[1,2,3]). Restore ckpt2 over the
    # live update-2 objects — same wiring, same identity — and run
    # update 3. ──────────────────────────────────────────────────────
    restored2 = restore_training_state(
        ckpt2, policy, critic, lora_info, opt_a, opt_c,
        runtime_identity=resume_identity_for(
            contract, loop_config, trainer_config, runtime,
            configure_numerical_mode(FORMAL_NUMERICAL_MODE), policy,
            base_seed=1,
            n_envs=int(adapter.n_envs),
            lora_info=lora_info,
            actor_optimizer=opt_a,
            critic_optimizer=opt_c,
        ),
        task_sampler=sampler,
    )
    assert restored2["ppo_update_count"] == 2, restored2
    assert restored2["outer_iteration"] == 2, restored2

    loop_config3 = LoopConfig(
        mode="one_update", n_episodes=32, max_ep_len=60, device=device,
        output_dir=out_dir,
    )
    result3 = run_e2e_batch(
        adapter, policy, critic, contract, loop_config3, trainer_config,
        specs, opt_a, opt_c, lora_info, runtime, resume_from=ckpt2,
        task_sampler=sampler,
    )
    ckpt3 = result3["checkpoint_path"]
    assert ckpt3 is not None, "resumed update 3 must write a checkpoint"
    saved3 = CheckpointBuilder.load(ckpt3)
    assert saved3.ppo_update_count == 3, saved3.ppo_update_count
    assert saved3.outer_iteration == 3, saved3.outer_iteration
    out.update({
        "metrics3": _jsonable(result3["metrics"]),
        "env_steps3": int(result3["env_steps"]),
        "sampler_state3": _jsonable(sampler.state_dict()),
        "ckpt3": {
            "ppo_update_count": saved3.ppo_update_count,
            "outer_iteration": saved3.outer_iteration,
            "optimizer_step_count": saved3.optimizer_step_count,
            "environment_transition_count":
                saved3.environment_transition_count,
            "completed_episode_count": saved3.completed_episode_count,
            "lora_checksum": lora_checksum_of(saved3.lora_state),
        },
    })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("CHILD RESUME OK")
    print("CHILD RESUME2 OK")


if __name__ == "__main__":
    main()
"""


def _compare_value(parent_value: Any, child_value: Any, path: str,
                   mismatches: List[str]) -> None:
    """Recursive two-state structural comparison (R6-C §11.4.1).

    Handles every value shape the trainer metrics dict can carry:
    scalars (isclose), None (presence-compare), dicts (recurse over
    the key union — a key present on one side only is a structural
    divergence), and lists (element-wise). Never coerces a non-number
    through float(): dict leaves (OptionalStatistic records, per-role
    maps), str leaves (invalid_reason, hashes, variance space) and
    list leaves (action_variance_by_dim) are legitimate metric shapes.
    (GPU RED 2026-08-18: the flat two-level loop crashed with
    float(None) on one-sided nulls, then float(dict) on the nested
    executed_vs_unexecuted_grad_norm.ratio / outcome_counts role maps
    once the None case was handled; str and list leaves were the next
    latent class — one recursive walker covers them all.) Both None ==
    equal; one-sided None == genuine divergence -> mismatch. The
    legacy NaN-sentinel skip is retained: non-finite numerics are an
    engine violation the strict writer catches at persist time.
    """
    if parent_value is None or child_value is None:
        if parent_value != child_value:
            mismatches.append(
                f"{path}: None vs {child_value!r}"
                if parent_value is None else
                f"{path}: {parent_value!r} vs None")
        return
    if isinstance(parent_value, dict) or isinstance(child_value, dict):
        if not (isinstance(parent_value, dict)
                and isinstance(child_value, dict)):
            mismatches.append(
                f"{path}: dict vs {type(child_value).__name__}")
            return
        for key in parent_value.keys() | child_value.keys():
            if key not in parent_value:
                mismatches.append(
                    f"{path}.{key}: <missing> vs {child_value[key]!r}")
                continue
            if key not in child_value:
                mismatches.append(
                    f"{path}.{key}: {parent_value[key]!r} vs <missing>")
                continue
            _compare_value(parent_value[key], child_value[key],
                           f"{path}.{key}", mismatches)
        return
    if isinstance(parent_value, (list, tuple)) or isinstance(
            child_value, (list, tuple)):
        if not (isinstance(parent_value, (list, tuple))
                and isinstance(child_value, (list, tuple))):
            mismatches.append(
                f"{path}: list vs {type(child_value).__name__}")
            return
        if len(parent_value) != len(child_value):
            mismatches.append(
                f"{path}: length {len(parent_value)} vs "
                f"{len(child_value)}")
            return
        for i, (pv, cv) in enumerate(zip(parent_value, child_value)):
            _compare_value(pv, cv, f"{path}[{i}]", mismatches)
        return
    # R6-C rework (P1-4): non-finite on EITHER side is a hard-fail —
    # NaN/Inf is not a legitimate "no data" representation under the
    # two-state contract (no data = OptionalStatistic count==0 with
    # value None). The old parent-side skip let engine violations
    # through as equality.
    if isinstance(parent_value, (int, float)) and not np.isfinite(
            parent_value):
        mismatches.append(f"{path}: parent non-finite {parent_value!r}")
        return
    if isinstance(child_value, (int, float)) and not np.isfinite(
            child_value):
        mismatches.append(f"{path}: child non-finite {child_value!r}")
        return
    if isinstance(parent_value, (int, float)) and isinstance(
            child_value, (int, float)):
        if not math.isclose(
            float(child_value), float(parent_value),
            rel_tol=RESUME_RTOL, abs_tol=RESUME_ATOL,
        ):
            mismatches.append(f"{path}: {parent_value} vs {child_value}")
        return
    # Non-numeric scalars (str tags, hashes, ...): strict equality — a
    # differing tag is a genuine parity break.
    if parent_value != child_value:
        mismatches.append(f"{path}: {parent_value!r} vs {child_value!r}")


def _compare_metrics(parent: Dict[str, Any], child: Dict[str, Any],
                     path: str) -> List[str]:
    """Resume-parity comparison of the trainer metrics dicts (the G6
    gate-9 loop), including the GAE hashes compared explicitly.

    Both sides are canonicalized through _jsonable: the child crosses a
    JSON boundary, so its dict keys are strings — the parent's
    int-keyed dicts (per_denoise_approx_kl / per_denoise_clipfrac,
    keyed by the denoising step index) must be canonicalized the same
    way, or every int key raises KeyError (rework #3 gate-1 finding:
    KeyError: 0 before this fix).

    R6-C (§11.4.1): the metrics dict carries mixed value shapes —
    scalars, OptionalStatistic records (dict), per-role/per-index maps
    (dict), distribution summaries (dict), a list (action_variance_
    by_dim) and str tags — compared structurally by _compare_value:
    parity breaks are reported as mismatches, never float() coercion
    of a non-number (GPU RED 2026-08-18: float(None) round 2,
    float(dict) round 3, g6 gate-9; same latent class in the r3 gate).
    """
    parent = _jsonable(parent)
    mismatches: List[str] = []
    for k, v in parent.items():
        if k not in child:
            mismatches.append(f"{path}.{k}: <missing> in child")
            continue
        _compare_value(v, child[k], f"{path}.{k}", mismatches)
    for k in child:
        if k not in parent:
            mismatches.append(f"{path}.{k}: <missing> in parent")
    return mismatches


@pytest.mark.gpu
def test_gpu_fresh_process_resume_chain(tmp_path):
    """Real GPU: update1 -> save -> FRESH PROCESS resume -> update2,
    with update-2 metrics/counters/sampler cursors agreeing with the
    continuous world; schema-v4 checkpoint facts pinned on the real save.
    """
    manifest, worlds, _, _, bc_ckpt = _assets()

    # ── World A (continuous): update 1 -> ckpt1 -> update 2 ──────────
    policyA, loraA, criticA, optAa, optAc = _fresh_world(DEVICE, bc_ckpt)
    adapterA, specsA, contract = _train_adapter(manifest, worlds)
    runtime = {
        "device": DEVICE,
        "reward_version": "v16",
        "manifest_hash": "r3-rework3-gpu-gate",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "world_assets_hash": "?",
        "timestamp": "r3-rework3-gpu-gate",
    }
    loop1 = LoopConfig(
        mode="one_update", n_episodes=EPISODES, max_ep_len=MAX_EP_LEN,
        device=DEVICE, output_dir=str(tmp_path / "parent_run1"),
    )
    trainer = TrainerConfig(batch_size=BATCH, device=DEVICE)
    samplerA = EpisodeTaskSampler(
        specsA, DEFAULT_ROLE_TARGETS, base_seed=SAMPLER_BASE_SEED
    )
    res1 = run_e2e_batch(
        adapterA, policyA, criticA, contract, loop1, trainer, specsA,
        optAa, optAc, loraA, runtime, task_sampler=samplerA,
    )
    ckpt1 = res1["checkpoint_path"]
    assert ckpt1 is not None, "update 1 must write a checkpoint"

    # ── schema-v4 checkpoint facts on the REAL save ──────────────────
    saved1 = CheckpointBuilder.load(ckpt1)
    ident1 = saved1.lora_identity
    assert saved1.schema_version == "5", saved1.schema_version
    assert ident1["n_lora_param_tensors"] == 2 * ident1["n_adaptors"], (
        "real insert_lora creates A+B per adaptor — param-tensor count "
        f"{ident1['n_lora_param_tensors']} != 2 * n_adaptors "
        f"{ident1['n_adaptors']} (P0-1)"
    )
    assert ident1["lora_checksum"] == lora_checksum_of(saved1.lora_state), (
        "identity checksum must bind the saved tensors"
    )
    assert ident1["lora_checksum"] != EMPTY_LORA_CHECKSUM
    env_rng = saved1.environment_rng_state
    assert env_rng["restorable"] is True, env_rng
    assert env_rng["python_hash_seed"] == os.environ.get(
        "PYTHONHASHSEED", ""
    ), env_rng
    assert saved1.sampler_state, "checkpoint must carry the sampler state"
    assert saved1.ppo_update_count == 1
    assert saved1.environment_transition_count == res1["env_steps"]
    ckpt1_transitions = saved1.environment_transition_count

    # ── continuous update 2 (World A keeps going) ────────────────────
    loop2 = LoopConfig(
        mode="one_update", n_episodes=EPISODES, max_ep_len=MAX_EP_LEN,
        device=DEVICE, output_dir=str(tmp_path / "parent_run2"),
    )
    res2 = run_e2e_batch(
        adapterA, policyA, criticA, contract, loop2, trainer, specsA,
        optAa, optAc, loraA, runtime, resume_from=ckpt1,
        task_sampler=samplerA,
    )
    m2 = res2["metrics"]
    ckpt2A = res2["checkpoint_path"]
    assert ckpt2A is not None
    saved2A = CheckpointBuilder.load(ckpt2A)
    assert saved2A.ppo_update_count == 2, saved2A.ppo_update_count
    samplerA2 = _jsonable(samplerA.state_dict())

    # ── continuous update 3 (Guide R3.2 Step 4: [1,2,3] witnessed) ────
    loop3 = LoopConfig(
        mode="one_update", n_episodes=EPISODES, max_ep_len=MAX_EP_LEN,
        device=DEVICE, output_dir=str(tmp_path / "parent_run3"),
    )
    res3 = run_e2e_batch(
        adapterA, policyA, criticA, contract, loop3, trainer, specsA,
        optAa, optAc, loraA, runtime, resume_from=ckpt2A,
        task_sampler=samplerA,
    )
    m3 = res3["metrics"]
    ckpt3A = res3["checkpoint_path"]
    assert ckpt3A is not None, "update 3 must write a checkpoint"
    saved3A = CheckpointBuilder.load(ckpt3A)
    assert saved3A.ppo_update_count == 3, saved3A.ppo_update_count
    assert saved3A.outer_iteration == 3, saved3A.outer_iteration
    samplerA3 = _jsonable(samplerA.state_dict())

    # free the card before the child process starts
    _release(policyA, criticA, optAa, optAc, adapterA, loraA,
             res1, res2, res3)

    # ── fresh-process resume (World B) ───────────────────────────────
    child_dir = tmp_path / "child"
    child_dir.mkdir(parents=True, exist_ok=True)
    child_script = child_dir / "resume_child.py"
    child_script.write_text(_CHILD_SCRIPT, encoding="utf-8")
    out_json = str(child_dir / "child_evidence.json")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # text=True without encoding decodes with the locale default (ASCII on
    # this box) — the child logs a UTF-8 "λ" and would crash the decode
    # AFTER the child succeeded, hiding the real result. Pin UTF-8.
    proc = subprocess.run(
        [
            sys.executable, str(child_script), str(REPO_ROOT), ckpt1,
            manifest, worlds, bc_ckpt, str(child_dir / "run"), out_json,
        ],
        env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert proc.returncode == 0, (
        f"child resume failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    with open(out_json, encoding="utf-8") as f:
        child = json.load(f)
    assert "CHILD RESUME OK" in proc.stdout

    # ── parity: resumed update 2 == continuous update 2 ──────────────
    mismatches = _compare_metrics(m2, child["metrics"], "metrics")
    for key in ("advantages_hash", "returns_hash"):
        if m2.get(key) != child["metrics"].get(key):
            mismatches.append(
                f"{key}: {m2.get(key)} vs {child['metrics'].get(key)}"
            )
    assert not mismatches, (
        "fresh-process update-2 metrics deviate from the continuous "
        "world: " + "; ".join(mismatches[:8])
    )
    # counters: child's checkpoint must carry the same cumulative facts
    assert child["ckpt2"]["ppo_update_count"] == 2
    assert child["ckpt2"]["environment_transition_count"] == (
        ckpt1_transitions + child["env_steps"]
    ), "resumed counter must continue from the checkpoint's"
    for key in ("ppo_update_count", "outer_iteration",
                "optimizer_step_count", "environment_transition_count",
                "completed_episode_count"):
        assert child["ckpt2"][key] == getattr(saved2A, key), (
            f"counter {key}: child {child['ckpt2'][key]} vs continuous "
            f"{getattr(saved2A, key)}"
        )
    # sampler cursors: the fresh process continued the checkpoint's
    # attempt/cursor state — post-update-2 cursors must match exactly
    assert child["sampler_state"] == samplerA2, (
        "fresh-process sampler cursors differ from the continuous world's"
    )

    # ── second resume parity: resumed update 3 == continuous update 3 ──
    assert "CHILD RESUME2 OK" in proc.stdout
    mismatches3 = _compare_metrics(m3, child["metrics3"], "metrics3")
    for key in ("advantages_hash", "returns_hash"):
        if m3.get(key) != child["metrics3"].get(key):
            mismatches3.append(
                f"{key}: {m3.get(key)} vs {child['metrics3'].get(key)}"
            )
    assert not mismatches3, (
        "fresh-process update-3 metrics deviate from the continuous "
        "world: " + "; ".join(mismatches3[:8])
    )
    assert child["ckpt3"]["ppo_update_count"] == 3
    assert child["ckpt3"]["outer_iteration"] == 3
    assert child["ckpt3"]["environment_transition_count"] == (
        child["ckpt2"]["environment_transition_count"]
        + child["env_steps3"]
    ), "second-resume counter must continue from the checkpoint's"
    for key in ("ppo_update_count", "outer_iteration",
                "optimizer_step_count", "environment_transition_count",
                "completed_episode_count"):
        assert child["ckpt3"][key] == getattr(saved3A, key), (
            f"counter {key} (update 3): child {child['ckpt3'][key]} vs "
            f"continuous {getattr(saved3A, key)}"
        )
    assert child["sampler_state3"] == samplerA3, (
        "second-resume sampler cursors differ from the continuous world's"
    )
    print("R3 rework #3 gate 1 PASS: fresh-process resume == continuous "
          f"({len(child['metrics'])} metric groups, 2 resumes, "
          "outer_iteration [1,2,3])")


# ═══════════════════════════════════════════════════════════════════
# Gate 2: canonical paired eval on the real trained LoRA policy
# ═══════════════════════════════════════════════════════════════════

def _git_rev(rev: str) -> str:
    out = subprocess.run(
        ["git", "rev-parse", rev], cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    assert out.returncode == 0, f"git rev-parse {rev} failed: {out.stderr}"
    return out.stdout.strip()


def _real_runtime_facts(manifest: str, bc_ckpt: str) -> Dict[str, Any]:
    """The eval CLI validates the checkpoint against the PREFLIGHT's real
    hashes — the training runtime must record the same real facts."""
    return {
        "official_dppo_commit": OFFICIAL_DPPO_COMMIT,
        "project_git_commit": _git_rev("HEAD"),
        "project_tree_hash": _git_rev("HEAD^{tree}"),
        "contract_hash": contract_hash_of(RLSystemContract()),
        "manifest_hash": file_sha256(Path(manifest)),
        "bc_checkpoint_hash": file_sha256(Path(bc_ckpt)),
        "world_assets_hash": world_assets_hash_of(
            Path(DEFAULT_WORLDS_TRAIN),
            [e["world_uid"] for e in manifest_train_entries(manifest)]),
        "source_file_hashes": official_source_file_hashes(REPO_ROOT),
    }


def _bank_specs(specs: Sequence[Any], max_per_role: int = 2) -> List[Any]:
    """Deterministic bank selection: up to max_per_role specs per role,
    all three capability roles covered (the artifact's capability
    metrics hard-fail on a role with no record)."""
    by_role: Dict[str, List[Any]] = {}
    for s in specs:
        by_role.setdefault(s.role, []).append(s)
    chosen: List[Any] = []
    for role in CAPABILITY_ROLES:
        chosen.extend(by_role.get(role, [])[:max_per_role])
    assert len(chosen) >= 3, (
        "eval pool must cover the capability roles, got roles: "
        f"{sorted(by_role)}"
    )
    return chosen


@pytest.mark.gpu
def test_gpu_canonical_paired_eval_real_artifact(tmp_path):
    """Real GPU: train one update with real identity facts, freeze a
    paired bank on the real eval pool, run the official eval entry as a
    subprocess, and verify the persisted artifact's bindings."""
    manifest, worlds, worlds_val, worlds_test, bc_ckpt = _assets()

    # ── train one real update via the CANONICAL train entry (production
    #    main(): preflight + real identity + release on the final batch —
    #    the exact path formal training runs; no helper-level wiring,
    #    no placeholder identity) ──────────────────────────────────────
    train_out = tmp_path / "train"
    train_script = REPO_ROOT / "scripts" / "train_dppo_uav_official.py"
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    train_proc = subprocess.run(
        [
            sys.executable, str(train_script),
            "--stage", "one_update",
            "--bc_ckpt", bc_ckpt,
            "--admission_manifest", manifest,
            "--worlds_train", worlds,
            "--output_dir", str(train_out),
            "--device", DEVICE,
            "--n_episodes", str(EPISODES),
            "--max_ep_len", str(MAX_EP_LEN),
            "--base_seed", str(SAMPLER_BASE_SEED),
            "--eval_n_episodes", "0",
        ],
        env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert train_proc.returncode == 0, (
        f"train CLI exit {train_proc.returncode}\n"
        f"stdout:\n{train_proc.stdout}\nstderr:\n{train_proc.stderr}"
    )
    ckpts = sorted(train_out.glob("dppo_uav_envstep_*.pt"))
    assert len(ckpts) == 1, [p.name for p in train_out.iterdir()]
    ckpt = str(ckpts[0])
    saved = CheckpointBuilder.load(ckpt)
    assert saved.checkpoint_kind == "release", (
        "the canonical train entry must materialize a RELEASE on the "
        "final batch — the eval cold-loads exactly this file"
    )
    ckpt_lora_checksum = lora_checksum_of(saved.lora_state)
    assert ckpt_lora_checksum != EMPTY_LORA_CHECKSUM

    # ── freeze the paired bank on a dedicated n_envs=1 eval adapter ──
    # (the canonical train entry built its own contract internally; the
    # bank/eval section needs the RL contract directly)
    contract = RLSystemContract()
    entries = manifest_eval_entries(manifest)
    specs_eval, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e))
         for e in entries]
    )
    pool = build_routea_eval_pool(
        manifest, worlds, worlds_val, worlds_test)
    records_eval = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs_eval
    }
    env_f = build_adapter(
        contract, worlds_dir=worlds, pool=pool,
        manifest_records=records_eval, device=DEVICE,
        max_episode_steps=MAX_EP_LEN, n_envs=1,
    )
    bank_specs = _bank_specs(specs_eval)
    seeds = list(range(1, len(bank_specs) + 1))
    bank = freeze_paired_bank(env_f, bank_specs, seeds)
    _release(env_f)
    bank_dict = paired_bank_to_dict(
        bank, source_hash=file_sha256(Path(manifest)),
        creation_commit="r3-rework3-gpu-gate",
    )
    bank_id = bank_dict["bank_id"]
    assert bank_id, "frozen bank must carry the v3 provenance bank_id"
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps(bank_dict, indent=2), encoding="utf-8"
    )
    density = OfflineDensity(counts={
        (0, 0, 0): 20, (1, 0, 0): 5, (0, 1, 0): 5, (0, 0, 1): 2,
        (1, 1, 1): 2, (2, 0, 0): 1,
    })
    density_path = tmp_path / "density.json"
    density_path.write_text(
        json.dumps(density.to_dict(), indent=2), encoding="utf-8"
    )

    # ── official eval entry (subprocess: preflight + identity + CLI) ─
    eval_out = tmp_path / "eval"
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    eval_script = REPO_ROOT / "scripts" / "eval_dppo_uav_official.py"
    proc = subprocess.run(
        [
            sys.executable, str(eval_script),
            "--checkpoint", ckpt,
            "--paired_bank", str(bank_path),
            "--offline_density", str(density_path),
            "--worlds_train", worlds,
            "--worlds_val", worlds_val,
            "--worlds_test", worlds_test,
            "--output_dir", str(eval_out),
            "--device", DEVICE,
        ],
        env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert proc.returncode == 0, (
        f"eval CLI exit {proc.returncode}\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    artifacts = sorted(eval_out.glob("paired_eval_*.json"))
    assert len(artifacts) == 1, [p.name for p in eval_out.iterdir()]
    with open(artifacts[0], encoding="utf-8") as f:
        artifact = json.load(f)

    # ── artifact bindings: frozen bank / LoRA checksum / hashes / trace
    assert artifact["checkpoint_hash"] == file_sha256(Path(ckpt))
    assert artifact["lora_checksum"] == ckpt_lora_checksum, (
        "artifact must bind the checkpoint's live LoRA tensors"
    )
    assert artifact["bank_id"] == bank_id
    assert artifact["offline_density_sha256"], "density provenance missing"
    assert artifact["use_obstacle_projection"] is False, (
        "§7.4: obstacle projection is policy-external and must be False"
    )
    assert artifact["observation_invalid_count"] == 0
    assert len(artifact["episodes"]) == len(bank)
    for i, (ep, entry) in enumerate(zip(artifact["episodes"], bank)):
        ec = ep["episode_config"]
        assert ec["episode_config_hash"] == entry.episode_config_hash, (
            f"record {i}: frozen episode_config_hash drifted"
        )
        assert ec["manifest_entry_hash"] == entry.manifest_entry_hash, (
            f"record {i}: frozen manifest hash drifted"
        )
        assert ep["world"] == entry.world_uid
        assert ep["role"] == entry.role
        assert ep["seed"] == entry.seed
        for side in ("bc", "rl"):
            trace = ep["trace"][side]
            assert trace, f"record {i} {side} trace is empty"
            for step in trace:
                assert all(k in step for k in TRACE_STEP_REQUIRED_KEYS), (
                    f"record {i} {side} trace step misses required keys"
                )
    for group in ("anticipatory_avoidance", "fullspace_support_expansion",
                  "dense_obstacle_ceiling"):
        assert artifact["capability_metrics"][group], (
            f"capability group {group} empty — capability role not covered"
        )
    print("R3 rework #3 gate 2 PASS: canonical paired eval artifact "
          f"({len(artifact['episodes'])} paired episodes) bound to "
          f"bank {bank_id[:12]} + ckpt LoRA {artifact['lora_checksum'][:12]}")


@pytest.mark.gpu
def test_gpu_canonical_budget_composition(tmp_path):
    """R5.8-A canonical composition (ruling §8.3 #1): the canonical train
    entry with --target_transitions 25 --rollout_transitions_per_update 8
    must collect exactly [8,8,8,1] across four batches and materialize a
    release — the P0-1 startup-ordering regression (UnboundLocalError) is
    caught here at the CLI level, not by loop-level tests."""
    manifest, worlds, _, _, bc_ckpt = _assets()
    train_out = tmp_path / "budget"
    train_script = REPO_ROOT / "scripts" / "train_dppo_uav_official.py"
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    proc = subprocess.run(
        [
            sys.executable, str(train_script),
            "--stage", "g7_micro",
            "--bc_ckpt", bc_ckpt,
            "--admission_manifest", manifest,
            "--worlds_train", worlds,
            "--output_dir", str(train_out),
            "--device", DEVICE,
            "--n_episodes", "32",
            "--max_ep_len", "100",
            "--base_seed", str(SAMPLER_BASE_SEED),
            "--eval_n_episodes", "0",
            "--target_transitions", "25",
            "--rollout_transitions_per_update", "8",
        ],
        env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert proc.returncode == 0, (
        f"budget train CLI exit {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # Retention keeps the latest 2 checkpoints by design (R5.8-D) — the
    # truthful completion signature is the metrics.jsonl cumulative
    # stream ([8,16,24,25]) + the final release checkpoint.
    records = [
        json.loads(line) for line in
        (train_out / "metrics.jsonl").read_text().splitlines() if line
    ]
    assert len(records) == 4, records
    assert [r["cumulative_env_steps"] for r in records] == [8, 16, 24, 25]
    final = CheckpointBuilder.load(
        str(sorted(train_out.glob("dppo_uav_envstep_*.pt"))[-1])
    )
    assert final.checkpoint_kind == "release", (
        "the budget-completing batch must materialize the release"
    )
    assert final.environment_transition_count == 25
