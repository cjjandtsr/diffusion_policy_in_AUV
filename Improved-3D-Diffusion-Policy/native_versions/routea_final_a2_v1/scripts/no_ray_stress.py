"""no_ray_stress.py — observation-only no-ray stress (R2.5 Step 4, R2 rework P1-4).

Requirement (guide R2.5 Step 4 + 审核裁决 P1-4): at least 10,000 ROLLOUT
frames through the REAL production adapter path (frozen env -> synth
wrapper -> UAVVectorEnvAdapter info -> gate), invalid=0, before G5/G7.
The R2.5 original evidence ran 9,000 wrapper-level frames + 1,000 adapter
steps — the audit ruling requires the FULL 10,000 to be adapter/rollout
frames, and the harness must be versioned (scripts/) with evidence bound
to the exact command, exit code, and tree hash that produced it.

The stress explicitly samples the G5 degenerate band (z > far*sin(8°+23.41°)
~= 5.21 m, world edge, outward yaw) — the family that previously voided
episodes — via the same mechanism part B used: teleport the frozen env's
drone pose before stepping, reset on the production cadence (a done slot
is reset_at()-ed before the next step).

Pass condition: total_frames >= 10,000 AND invalid == 0 AND far-dome
frames > 0 AND every far-dome frame carries a passing analytic
RayGeometryReport (report_failures == 0). Evidence JSON written to
--output; exit code 0/1 mirrors pass.

Run (after `git add` so the tree hash is complete):
  python scripts/no_ray_stress.py --frames 10000 --seed 0 \
      --output program/reports/routeA_rl_env_v1/g3_gate/NO_RAY_STRESS_EVIDENCE.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion_policy_3d.model.common.synth_pointcloud import (
    CAM_MOUNT_X,
    CAM_MOUNT_Y,
    CAM_MOUNT_Z,
    DepthCameraIntrinsics,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    PointCloudValidityGate,
    build_observation_window_mapping,
    validate_policy_observation_window,
)

CAM_FAR = DepthCameraIntrinsics().far
# G5 degenerate band: z > 10.0 * sin(8° + 23.41°) ~= 5.21 m with nothing
# within 10 m (world edge, outward yaw).
BAND_X = (-10.7, -10.1)
BAND_Y = (-4.6, -4.3)
BAND_Z = (5.25, 5.6)
OBSTACLE_WORLD = [
    {"name": "c1", "shape": "cylinder", "x": 0.0, "y": 0.0,
     "radius": 1.2, "height": 4.0},
    {"name": "b1", "shape": "box", "x": 3.0, "y": -2.0,
     "size_x": 2.0, "size_y": 2.0, "size_z": 3.0},
    {"name": "c2", "shape": "cylinder", "x": -4.0, "y": 5.0,
     "radius": 0.9, "height": 3.0},
]
RESET_CADENCE = 25  # periodic reset, production-style slot lifecycle
BAND_CADENCE = 5    # every 5th step drives the degenerate band


def _write_world(path: str, name: str, obstacles: list) -> str:
    jp = os.path.join(path, f"{name}.obstacles.json")
    with open(jp, "w") as f:
        json.dump({"world_id": name, "obstacles": obstacles}, f)
    return jp


def _sample_pose(rng: np.random.Generator, in_band: bool) -> np.ndarray:
    """Full-domain pose, or the G5 degenerate band with outward yaw."""
    if in_band:
        x = rng.uniform(*BAND_X)
        y = rng.uniform(*BAND_Y)
        z = rng.uniform(*BAND_Z)
        yaw = np.pi  # outward from the west edge
    else:
        x = rng.uniform(-14.0, 14.0)
        y = rng.uniform(-14.0, 14.0)
        z = rng.uniform(1.0, 5.2)
        yaw = rng.uniform(-np.pi, np.pi)
    return np.array([x, y, z], dtype=np.float32), float(yaw)


def _hit_count(points: np.ndarray, cam_far: float) -> int:
    """Distance-based mirror of environment_adapter._hit_count (R2.5)."""
    cam_body = np.array([CAM_MOUNT_X, CAM_MOUNT_Y, CAM_MOUNT_Z],
                        dtype=np.float32)
    dist = np.linalg.norm(np.asarray(points, dtype=np.float32) - cam_body,
                          axis=1)
    return int(np.count_nonzero(dist < cam_far - 0.1))


def run_stress(frames: int, seed: int, world_jsons: List[str]) -> Dict:
    """10k-scale rollout-frame stress through the real adapter path.

    Two one-slot adapters (empty / cluttered world) are stepped in
    interleave; each step's observation goes through the SAME pipeline
    production uses: frozen env -> ValidatedSynthAdapter -> adapter info
    (hit_count + ray_report) -> PointCloudValidityGate. Pose teleports
    drive the degenerate band every BAND_CADENCE steps; slots reset on a
    fixed cadence and whenever they go terminal (production semantics).
    """
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        EpisodeTaskSpec,
        UAVVectorEnvAdapter,
        manifest_record_key,
    )

    contract = RLSystemContract()
    gate = PointCloudValidityGate(contract, phase="unit")
    adapters = []
    specs = []
    for i, jp in enumerate(world_jsons):
        uid = f"env_stress{i}"
        adapter = UAVVectorEnvAdapter(
            contract, [(uid, jp)], device="cpu", max_episode_steps=100,
            # E040 formal training default (--max_ep_len 100); aligned with
            # the production train config (blindspot audit 2026-08-10).
            manifest_records={
                manifest_record_key(uid, "anticipatory", "a"): "mhash-a",
            },
        )
        adapters.append(adapter)
        specs.append(EpisodeTaskSpec(
            world_uid=uid, obstacle_json="unused", role="anticipatory",
            split="rl_train", admission_id="a",
            generator_version="routeA_rl_v1.3", manifest_entry_hash="mhash-a",
        ))
    rng = np.random.default_rng(seed)
    stats = {"n_frames": 0, "invalid": 0, "invalid_reasons": [],
             "far_dome_frames": 0, "report_failures": 0,
             "resets": 0, "early_terminations": 0,
             "unique_min": None, "unique_max": None, "dist_err_max": 0.0}
    zero = np.zeros((1, contract.exec_steps, contract.action_dim),
                    dtype=np.float32)
    # per-adapter done flag: the worlds interleave, so a terminal flag from
    # world A must never suppress world B's reset (and vice versa)
    done_flags = [True] * len(adapters)
    for i in range(frames):
        k = i % len(adapters)
        adapter = adapters[k]
        env0 = adapter._envs[0]
        # production slot lifecycle: reset on the fixed cadence AND whenever
        # the slot went terminal (train_loop reset_at()s a done slot). The
        # env's default disturb (p=0.2, scale 0.3) can shove a band pose
        # above z_max+1=6.0 -> altitude_exceed; that is legitimate episode
        # termination, not a frame issue.
        if done_flags[k] or i % RESET_CADENCE == 0:
            # rework #3: reset_at only accepts done/voided slots — a cadence
            # reset hitting an ACTIVE slot must be explicitly discarded
            # (abort_episode: in-flight -> DONE without fault) before reset.
            if not done_flags[k]:
                adapter.abort_episode([0])
            obs, info = adapter.reset([specs[k]], seeds=[int(seed + i)])
            stats["resets"] += 1
            done_flags[k] = False
        pose, yaw = _sample_pose(rng, in_band=(i % BAND_CADENCE == 0))
        env0._drone_pos = pose
        env0._drone_yaw = yaw
        obs, reward, terminated, truncated, info = adapter.step(zero)
        stats["n_frames"] += 1
        done_flags[k] = bool(terminated[0] or truncated[0])
        if done_flags[k]:
            stats["early_terminations"] += 1
        window = validate_policy_observation_window(
            info[0].get("observation_window_mapping")
            or build_observation_window_mapping(
                obs, contract.obs_steps, CAM_FAR),
            obs_steps=contract.obs_steps,
            gate=gate,
        )
        if not window.valid:
            for _fr in window.frame_results:
                if not _fr.valid:
                    stats["invalid"] += 1
                    r = _fr.invalid_reason or "unknown"
                    if r not in stats["invalid_reasons"]:
                        stats["invalid_reasons"].append(r)
        rpt = info[0].get("ray_report")
        if rpt is not None:
            stats["far_dome_frames"] += 1
            if not rpt.passes:
                stats["report_failures"] += 1
            u = rpt.unique_count
            stats["unique_min"] = u if stats["unique_min"] is None else min(
                stats["unique_min"], u)
            stats["unique_max"] = u if stats["unique_max"] is None else max(
                stats["unique_max"], u)
            cam_body = np.array([CAM_MOUNT_X, CAM_MOUNT_Y, CAM_MOUNT_Z],
                                dtype=np.float32)
            derr = float(np.max(np.abs(
                np.linalg.norm(obs["point_cloud"][0, -1] - cam_body, axis=1)
                - CAM_FAR)))
            stats["dist_err_max"] = max(stats["dist_err_max"], derr)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{frames} frames, invalid={stats['invalid']}, "
                  f"far-dome={stats['far_dome_frames']}, resets="
                  f"{stats['resets']}", flush=True)
    return stats


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="no_ray_stress_")
    empty_json = _write_world(tmp, "env_empty", [])
    obst_json = _write_world(tmp, "env_obst", OBSTACLE_WORLD)

    print(f"no-ray stress: {args.frames} rollout frames (real adapter path,"
          f" two worlds), seed={args.seed}")
    stats = run_stress(args.frames, args.seed, [empty_json, obst_json])

    ok = (stats["n_frames"] >= 10000 and stats["invalid"] == 0
          and stats["far_dome_frames"] > 0 and stats["report_failures"] == 0)
    evidence = {
        "title": "R2.5 no-ray stress evidence (R2 rework P1-4: all frames"
                 " adapter/rollout path)",
        "pass": ok,
        "total_frames": stats["n_frames"],
        "invalid": stats["invalid"],
        "far_dome_frames": stats["far_dome_frames"],
        "report_failures": stats["report_failures"],
        "rollout_path": "UAVVectorEnvAdapter.reset/step -> info "
                        "(hit_count + ray_report) -> PointCloudValidityGate",
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "exit_code": 0 if ok else 1,
        "tree_hash": _tree_hash(),
        "stats": stats,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"\nSTRESS {'PASS' if ok else 'FAIL'}: "
          f"{stats['n_frames']} rollout frames, invalid={stats['invalid']}, "
          f"far-dome={stats['far_dome_frames']}, "
          f"report_failures={stats['report_failures']}")
    print(f"evidence: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
