"""Read-only, single-task initial raw-frame probe for first-H4 replay."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from scripts.routea_causal_diagnostics.reporting import (
    file_sha256,
    verify_sha256_manifest,
)


SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def derive_episode_seed(world_id, start_pos, goal_pos, reset_seed):
    """Mirror KinematicUAVEnv.reset's fixed-episode identity, without an env."""
    try:
        world = int(world_id)
    except (ValueError, TypeError):
        world = str(world_id)
    start = np.asarray(start_pos, dtype=np.float32)
    goal = np.asarray(goal_pos, dtype=np.float32)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start/goal must each have three coordinates")
    payload = json.dumps({
        "world_id": world,
        "goal_w": [round(float(value), 4) for value in goal],
        "drone_pos": [round(float(value), 4) for value in start],
        "seed": int(reset_seed),
    }, sort_keys=True).encode("utf-8")
    return (
        hashlib.sha256(payload).hexdigest(),
        int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little"),
    )


def sample_initial_raw_frame(synth, start_pos, start_yaw, episode_seed, identity):
    """Call precisely the v17 initial-frame synthesis branch, with no policy."""
    position = np.asarray(start_pos, dtype=np.float32).copy()
    position[2] = max(position[2], 0.05)
    _, meta = synth.generate_with_metadata(
        position, float(start_yaw), num_points=4096, add_noise=True,
        rng_seed=int(episode_seed), episode_identity_hash=identity,
        decision_index=0,
    )
    return meta.raw_array_hash, meta.valid_hit_rays, meta.total_rays


def _parent_rows(path, task_id, repeat):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"task_id": "{}"'.format(task_id) not in line:
                continue
            row = json.loads(line)
            if (row["task_id"], int(row["repeat"]), int(row["decision_index"])) != (
                task_id, repeat, 0,
            ):
                continue
            lane = row["lane"]
            if lane in ("u0", "seed3-u32"):
                if lane in rows:
                    raise ValueError("duplicate parent initial row: " + lane)
                rows[lane] = row
    if set(rows) != {"u0", "seed3-u32"}:
        raise ValueError("missing paired parent initial rows")
    return rows


def probe(selection_dir, capture_dir, task_id, repeat, stopped_raw_hash):
    if not SHA256.fullmatch(stopped_raw_hash):
        raise ValueError("stopped raw hash must be lowercase SHA256")
    verify_sha256_manifest(selection_dir)
    verify_sha256_manifest(capture_dir)
    selection = json.loads((selection_dir / "selection.json").read_text(encoding="utf-8"))
    entries = [entry for pair in selection["pairs"] for entry in (pair["break"], pair["rescue"])
               if entry["task_id"] == task_id and int(entry["repeat"]) == repeat]
    if len(entries) != 1:
        raise ValueError("selected task/repeat is not unique")
    entry = entries[0]
    config = entry["episode_config"]
    geometry = Path(entry["geometry_path"])
    if file_sha256(geometry) != entry["geometry_sha256"]:
        raise ValueError("world geometry hash differs from selection")
    rows = _parent_rows(capture_dir / "paired_trace_transitions.jsonl", task_id, repeat)
    bc, rl = rows["u0"], rows["seed3-u32"]
    common = ("raw_frame_hash", "episode_identity_hash", "env_reset_seed", "action_seed",
              "position_before_xyz", "yaw_before")
    if any(bc[key] != rl[key] for key in common):
        raise ValueError("parent initial rows disagree")
    if int(bc["env_reset_seed"]) != int(entry["env_reset_seed"]):
        raise ValueError("reset seed differs from selection")
    if not np.array_equal(np.asarray(config["start_pos"], dtype=np.float32),
                          np.asarray(bc["position_before_xyz"], dtype=np.float32)):
        raise ValueError("initial position differs from selection")
    if float(config["start_yaw"]) != float(bc["yaw_before"]):
        raise ValueError("initial yaw differs from selection")
    if config["world_name"] != entry["world"]:
        raise ValueError("world differs from selection")
    identity, episode_seed = derive_episode_seed(
        entry["world"], config["start_pos"], config["goal_pos"], entry["env_reset_seed"]
    )
    if identity != bc["episode_identity_hash"]:
        raise ValueError("episode identity differs from parent")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for parent-device comparison")
    synth = SynthPointCloud(str(geometry), device="cuda:0")
    first = sample_initial_raw_frame(synth, config["start_pos"], config["start_yaw"],
                                     episode_seed, identity)
    torch.manual_seed(987654321)
    second = sample_initial_raw_frame(synth, config["start_pos"], config["start_yaw"],
                                      episode_seed, identity)
    return {
        "status": "RAW_FRAME_PROBE_COMPLETE",
        "task_id": task_id,
        "repeat": repeat,
        "parent_raw_frame_hash": bc["raw_frame_hash"],
        "stopped_raw_frame_hash": stopped_raw_hash,
        "probe_raw_frame_hashes": [first[0], second[0]],
        "within_process_exact": first == second,
        "matches_parent": [sample[0] == bc["raw_frame_hash"] for sample in (first, second)],
        "matches_stopped": [sample[0] == stopped_raw_hash for sample in (first, second)],
        "valid_hit_rays": [first[1], second[1]],
        "total_rays": [first[2], second[2]],
        "episode_identity_hash": identity,
        "env_reset_seed": int(entry["env_reset_seed"]),
        "episode_rng_seed": episode_seed,
        "action_seed_parent_only": int(bc["action_seed"]),
        "start_pos_float32_hex": [float(value).hex() for value in
                                  np.asarray(config["start_pos"], dtype=np.float32)],
        "start_yaw_hex": float(config["start_yaw"]).hex(),
        "geometry_sha256": entry["geometry_sha256"],
        "selection_sha256": file_sha256(selection_dir / "selection.json"),
        "parent_trace_sha256": file_sha256(capture_dir / "paired_trace_transitions.jsonl"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "model_loaded": False,
        "policy_forward": False,
        "training_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--stopped-raw-hash", required=True)
    args = parser.parse_args()
    try:
        result = probe(args.selection_dir, args.capture_dir, args.task_id,
                       args.repeat, args.stopped_raw_hash)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "RAW_FRAME_PROBE_INPUT_ERROR", "detail": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
