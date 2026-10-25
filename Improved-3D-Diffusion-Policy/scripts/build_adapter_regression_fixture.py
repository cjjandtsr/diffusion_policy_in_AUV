"""build_adapter_regression_fixture.py — freeze the G4 replacement fixture.

Design §5.6 + Guide R4.1: the ORIGINAL Gazebo—kinematic equivalence
fixture was never found (ASSUMPTION-4), so this builds a REPLACEMENT
fixture (``replacement_adapter_non_regression_fixture``). It does NOT
re-prove Gazebo equivalence — it pins the frozen direct KinematicUAVEnv
v2 path (oracle) against the canonical UAVVectorEnvAdapter boundary
(candidate) over fixed inputs.

Per case (three deterministic action classes):
  1. build the candidate adapter (n_envs=1, production construction),
     reset slot 0 with a fixed (spec, seed) — the admission-sampled
     episode config is deterministic (np.random.RandomState(seed)), so it
     is RECORDED as the fixed initial state;
  2. build a fresh ORACLE (direct KinematicUAVEnv, frozen default synth
     factory, device-free CPU construction) and reset it with the exact
     recorded episode config + seed;
  3. step both paths with the SAME fixed (4, 11) action sequence and
     record per-step position/yaw/vel/agent_pos/pc-frame hashes/reward/
     termination flags.

The fixture is read-only after generation; its sha256 is bound by the
gate report (run_adapter_regression_gate.py), and both are committed.

Run:
  python scripts/build_adapter_regression_fixture.py \
      --out /home/cjj/AUV/program/reports/routeA_rl_env_v1/\
g4_adapter_regression/ADAPTER_REGRESSION_FIXTURE.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    UAVVectorEnvAdapter,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_task_specs,
    entry_metadata_hash,
    manifest_train_entries,
)

# The frozen E040 production path synthesizes on the default SynthPointCloud
# device (cuda:0 — train_ddpm_ppo_meanlogp_3_e040.py:715). Both sides of the
# fixture must mirror the frozen construction, so the oracle pool and the
# candidate adapter use the same default synth device.
ORACLE_DEVICE = "cuda:0"
CANDIDATE_DEVICE = "cuda:0"

DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
DEFAULT_OUT = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/g4_adapter_regression"
    "/ADAPTER_REGRESSION_FIXTURE.json"
)

ORACLE_SOURCES = [
    "diffusion_policy_3d/env/kinematic_uav_env_v2.py",
    "diffusion_policy_3d/env/traj_utils.py",
    "diffusion_policy_3d/env/collision_geom.py",
    "diffusion_policy_3d/model/common/synth_pointcloud.py",
]
CANDIDATE_SOURCES = [
    "diffusion_policy_3d/rl/dppo_uav/environment_adapter.py",
    "diffusion_policy_3d/rl/dppo_uav/integration.py",
    "diffusion_policy_3d/rl/dppo_uav/contracts.py",
    "diffusion_policy_3d/rl/dppo_uav/validated_synth_adapter.py",
    "diffusion_policy_3d/rl/dppo_uav/planner.py",
]

SEED_BASE = 7000
STEPS = 5
# Per-case seeds probed to survive the full 5 steps (straight on world
# 000001 collided for 7000-7017; 7018 was probed clean — the fixture is
# rebuilt whenever the worlds/manifest change, re-probing at build time).
CASE_SEEDS = {
    "straight_no_obstacle": 7018,
    "yaw_lateral": 7001,
    "z_composite": 7002,
}

# Three deterministic action classes (design §5.6 / Guide R4.1 Step 2):
# (4, 11) per step — 11d = [dpos_w(3), dvel_w(3), acc_w(3), dyaw, yaw_rate].
# Moderate deltas keep the episode near the admitted start; any early
# termination is recorded per step and must reproduce bitwise on replay.
def _straight_actions() -> np.ndarray:
    row = np.array([0.40, 0.0, 0.0, 0.40, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.stack([row] * 4)


def _yaw_lateral_actions() -> np.ndarray:
    row = np.array([0.15, 0.15, 0.0, 0.20, 0.20, 0.0,
                    0.0, 0.0, 0.0, 0.12, 0.12], dtype=np.float32)
    return np.stack([row] * 4)


def _z_composite_actions(start_z: float, z_mid: float) -> np.ndarray:
    # z direction chosen from the resolved start z to stay inside the
    # curriculum band [z_min, z_max] = [1.0, 5.0] (0.2 m total swing).
    dz = 0.05 if start_z <= z_mid else -0.05
    rows = [
        np.array([0.20, 0.15, dz, 0.25, 0.20, 0.2 * dz * 10,
                  0.04, -0.03, 0.02, -0.10, -0.10], dtype=np.float32),
        np.array([0.20, 0.15, dz, 0.25, 0.20, 0.2 * dz * 10,
                  -0.03, 0.04, 0.02, -0.10, -0.10], dtype=np.float32),
        np.array([0.20, 0.15, dz, 0.25, 0.20, 0.2 * dz * 10,
                  0.04, 0.02, -0.03, -0.10, -0.10], dtype=np.float32),
        np.array([0.20, 0.15, dz, 0.25, 0.20, 0.2 * dz * 10,
                  -0.02, 0.03, 0.04, -0.10, -0.10], dtype=np.float32),
    ]
    return np.stack(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(frame, dtype=np.float32).tobytes()
    ).hexdigest()


def _record_obs(adapter: UAVVectorEnvAdapter, obs: Dict[str, np.ndarray],
                n_envs: int, frames_dir: Path, tag: str) -> Dict[str, Any]:
    """Per-frame content hashes + hit counts over one env-level obs
    window; each expected frame is frozen as a .npy sidecar."""
    pcs = obs["point_cloud"][0] if n_envs else obs["point_cloud"]
    poses = obs["agent_pos"][0] if n_envs else obs["agent_pos"]
    hashes = []
    hits = []
    files = []
    for i in range(pcs.shape[0]):
        frame = np.ascontiguousarray(pcs[i], dtype=np.float32)
        hashes.append(_frame_sha256(frame))
        hits.append(int(adapter._hit_count(pcs[i])))
        fp = frames_dir / f"{tag}_f{i}.npy"
        np.save(fp, frame)
        files.append(str(fp))
    return {
        "agent_pos": poses.astype(np.float32).tolist(),
        "pc_frame_sha256": hashes,
        "hit_count": hits,
        "frame_files": files,
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--worlds-dir", default=DEFAULT_WORLDS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    entries = manifest_train_entries(args.manifest)
    if not entries:
        raise SystemExit("admission manifest has no admitted rl_train entries")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required: the frozen E040 path synthesizes on cuda:0 "
            "(SynthPointCloud default device); the fixture pins that path"
        )
    # Deterministic case-world selection: sorted uids, first three.
    chosen = sorted({e["world_uid"] for e in entries})[:3]
    if len(chosen) < 3:
        raise SystemExit(f"need 3 admitted worlds, got {len(chosen)}")

    specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries]
    )
    by_uid = {s.world_uid: s for s in specs}
    for uid in chosen:
        if uid not in by_uid:
            raise SystemExit(f"world {uid} missing from built specs")
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }

    contract = RLSystemContract()

    case_ids = ("straight_no_obstacle", "yaw_lateral", "z_composite")
    z_mid = 3.0  # curriculum C2 band midpoint (z_min=1.0, z_max=5.0)

    # Expected frames are frozen as .npy sidecars next to the fixture —
    # wiped on every rebuild so a stale frame never leaks into a new
    # fixture version.
    out_path = Path(args.out)
    frames_dir = out_path.parent / "ADAPTER_REGRESSION_FIXTURE_frames"
    import shutil

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cases: List[Dict[str, Any]] = []
    for i, uid in enumerate(chosen):
        spec = by_uid[uid]
        case_id = case_ids[i]
        seed = CASE_SEEDS[case_id]

        # ── candidate reset: a FRESH adapter per case (the gate replay
        #    builds fresh too). The admission-sampled config IS the fixed
        #    initial state (deterministic: np.random.RandomState(seed)) ──
        adapter = build_adapter(
            contract,
            worlds_dir=args.worlds_dir,
            manifest_path=args.manifest,
            manifest_records=records,
            device=CANDIDATE_DEVICE,  # production construction (conftest real_adapter)
            max_episode_steps=60,
            n_envs=1,
        )
        obs_c, info_c = adapter.reset_at([0], [spec], [seed])
        rec = adapter.current_episode_config(0)
        env_cfg = {
            "world_name": rec["world"],
            "goal_pos": [float(v) for v in rec["goal"]],
            "start_pos": [float(v) for v in rec["start"]],
            "start_yaw": float(rec["start_yaw"]),
            "disturb_events": [],
            "episode_id": rec["episode_id"],
        }

        # ── oracle: direct frozen KinematicUAVEnv with the same cfg.
        #    Construction mirrors the frozen E040 build_envs
        #    (train_ddpm_ppo_meanlogp_3_e040.py:715,736-739):
        #    synth_pc_pool of default-device SynthPointCloud instances,
        #    curriculum C2, reward v16, env defaults otherwise. ──
        world_file = Path(args.worlds_dir) / f"{uid}.world"
        obstacles_json = Path(args.worlds_dir) / f"{uid}.world.obstacles.json"
        synth_pc_pool = {uid: SynthPointCloud(
            obstacles_json=str(obstacles_json)
        )}
        oracle = KinematicUAVEnv(
            world_pool=[(uid, str(obstacles_json))],
            synth_pc_pool=synth_pc_pool,
            curriculum_stage="C2",
            max_episode_steps=60,
            reward_version="v16",
        )
        oracle.reset(seed=seed, options={"episode_config": env_cfg})
        # The ORACLE's reset obs is the expected value the candidate must
        # reproduce (the candidate's own reset obs is not authoritative).
        reset_obs = _record_obs(
            adapter, oracle._get_obs_dict(), n_envs=False,
            frames_dir=frames_dir, tag=f"{case_id}_reset",
        )

        if case_id == "straight_no_obstacle":
            actions = _straight_actions()
        elif case_id == "yaw_lateral":
            actions = _yaw_lateral_actions()
        else:
            actions = _z_composite_actions(
                float(rec["start"][2]), z_mid
            )
        actions_seq = np.stack([actions] * STEPS)

        expected: List[Dict[str, Any]] = []
        for k in range(STEPS):
            obs_o, reward, terminated, truncated, _ = oracle.step(
                actions.astype(np.float32)
            )
            exp = _record_obs(
                adapter, obs_o, n_envs=False,
                frames_dir=frames_dir, tag=f"{case_id}_step{k}",
            )
            exp.update({
                "position": oracle._drone_pos.copy().tolist(),
                "yaw": float(oracle._drone_yaw),
                "vel": oracle._drone_vel.copy().tolist(),
                "action_command": actions.tolist(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            expected.append(exp)
            if terminated or truncated:
                break
        # The frozen action sequence covers exactly the executed steps
        # (trimmed on early termination — the candidate must terminate at
        # the same step).
        actions_seq = actions_seq[: len(expected)]

        cases.append({
            "case_id": case_id,
            "world_uid": uid,
            "world_file": str(world_file),
            "world_hash": _sha256(world_file),
            "obstacle_json": str(obstacles_json),
            "obstacle_hash": _sha256(obstacles_json),
            "spec": {
                "world_uid": spec.world_uid,
                "obstacle_json": spec.obstacle_json,
                "role": spec.role,
                "split": spec.split,
                "admission_id": spec.admission_id,
                "generator_version": spec.generator_version,
                "manifest_entry_hash": spec.manifest_entry_hash,
            },
            "seed": seed,
            "steps_planned": STEPS,
            "initial_state": {
                "start": [float(v) for v in rec["start"]],
                "goal": [float(v) for v in rec["goal"]],
                "start_yaw": float(rec["start_yaw"]),
                "episode_config": env_cfg,
                "reset_obs": reset_obs,
            },
            "actions": actions_seq.tolist(),
            "expected": expected,
        })
        print(f"[{case_id}] world={uid} seed={seed} "
              f"start={rec['start']} steps_executed={len(expected)}"
              f" term={expected[-1]['terminated'] if expected else None}")

    fixture: Dict[str, Any] = {
        "schema_version": "g4-fixture-v1",
        "fixture_type": "replacement_adapter_non_regression_fixture",
        "fixture_id": "g4_adapter_regression_v1",
        "equivalence_fact": (
            "[FACT] Gazebo—kinematic equivalence was established before "
            "this fixture (design §5.6); the ORIGINAL equivalence fixture "
            "was never found (ASSUMPTION-4). This REPLACEMENT fixture does "
            "NOT re-prove Gazebo equivalence — it only pins the frozen "
            "direct KinematicUAVEnv v2 path against the canonical "
            "UAVVectorEnvAdapter boundary over fixed inputs."
        ),
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "tolerance_source": (
            "design §5.6 recorded threshold (original artifact unavailable; "
            "replacement_adapter_non_regression_fixture label per Guide R4.1"
        ),
        "pc_comparison_semantics": (
            "np.allclose(rtol=1e-5, atol=1e-6) per design §5.6 — NOT "
            "bitwise: the frozen CUDA synth pipeline (depth noise + "
            "dropout) is not bitwise-stable run-to-run (measured 60/60 "
            "repeat-run ULP deviations; the CPU synth is 0/60). Frame "
            "values are frozen as .npy sidecars with content sha256; the "
            "gate enforces the design threshold and records per-frame "
            "max_abs_diff."
        ),
        "frame_sidecar_dir": str(frames_dir),
        "created_commit": _git_head(),
        "oracle_device": ORACLE_DEVICE,
        "candidate_device": CANDIDATE_DEVICE,
        "oracle": {
            "name": "KinematicUAVEnv v2 direct (frozen E040 path)",
            "source_files": {
                rel: _sha256(REPO_ROOT / rel) for rel in ORACLE_SOURCES
            },
        },
        "candidate": {
            "name": "UAVVectorEnvAdapter (canonical boundary)",
            "source_files": {
                rel: _sha256(REPO_ROOT / rel) for rel in CANDIDATE_SOURCES
            },
        },
        "cases": cases,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2))
    print(f"fixture written: {out}")
    print(f"fixture sha256: {_sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
