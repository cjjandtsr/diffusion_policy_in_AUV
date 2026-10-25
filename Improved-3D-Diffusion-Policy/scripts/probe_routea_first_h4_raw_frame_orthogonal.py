"""Read-only orthogonal CUDA probe for one first-H4 initial raw frame."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
    canonical_array_hash,
)
from scripts.probe_routea_first_h4_raw_frame import (
    _parent_rows,
    derive_episode_seed,
)
from scripts.routea_causal_diagnostics.reporting import (
    file_sha256,
    verify_sha256_manifest,
)


TASK_ID = "env_012_obs22__cfg00"
REPEAT = 5
PARENT_RAW_HASH = "af3fc203486dc1a8b8516c3634719b7d963ec6deb37c7be3477ae4cc05da16a6"
STOPPED_RAW_HASH = "59d7a01ae51b74848d4df1b63126e47faaa184fbb146da304c0bd85058bae70a"
PRESEED = 987654321


def array_difference(first, second):
    a = np.asarray(first)
    b = np.asarray(second)
    if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape:
        raise ValueError("point arrays must share float32 shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("point arrays must be finite")
    different = a.view(np.uint32) != b.view(np.uint32)
    count = int(np.count_nonzero(different))
    return {
        "different_elements": count,
        "first_index": None if count == 0 else np.argwhere(different)[0].tolist(),
        "max_abs": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
    }


def sample_pair(synth, start_pos, start_yaw, episode_seed, identity, mode):
    if mode == "preseed":
        torch.manual_seed(PRESEED)
    elif mode == "prebuild-rays":
        synth._build_rays(synth.device)
    elif mode != "plain":
        raise ValueError("invalid probe mode")
    position = np.asarray(start_pos, dtype=np.float32).copy()
    position[2] = max(position[2], 0.05)

    def sample():
        return synth.generate_with_metadata(
            position, float(start_yaw), num_points=4096, add_noise=True,
            rng_seed=int(episode_seed), episode_identity_hash=identity,
            decision_index=0,
        )

    first, first_meta = sample()
    second, second_meta = sample()
    hashes = [first_meta.raw_array_hash, second_meta.raw_array_hash]
    return {
        "raw_frame_hashes": hashes,
        "within_process_exact": hashes[0] == hashes[1],
        "difference": array_difference(first, second),
        "ray_dirs_cam_hash": canonical_array_hash(
            synth._ray_dirs_cam.detach().cpu().numpy()
        ),
        "valid_hit_rays": [int(first_meta.valid_hit_rays), int(second_meta.valid_hit_rays)],
        "total_rays": [int(first_meta.total_rays), int(second_meta.total_rays)],
    }


def load_case(selection_dir, capture_dir):
    verify_sha256_manifest(selection_dir)
    verify_sha256_manifest(capture_dir)
    selection = json.loads((selection_dir / "selection.json").read_text(encoding="utf-8"))
    entries = [entry for pair in selection["pairs"]
               for entry in (pair["break"], pair["rescue"])
               if entry["task_id"] == TASK_ID and int(entry["repeat"]) == REPEAT]
    if len(entries) != 1:
        raise ValueError("fixed task/repeat is not unique")
    entry = entries[0]
    config = entry["episode_config"]
    geometry = Path(entry["geometry_path"])
    if file_sha256(geometry) != entry["geometry_sha256"]:
        raise ValueError("world geometry differs from selection")
    rows = _parent_rows(capture_dir / "paired_trace_transitions.jsonl", TASK_ID, REPEAT)
    bc, rl = rows["u0"], rows["seed3-u32"]
    common = ("raw_frame_hash", "episode_identity_hash", "env_reset_seed", "action_seed",
              "position_before_xyz", "yaw_before")
    if any(bc[key] != rl[key] for key in common):
        raise ValueError("paired parent initial rows disagree")
    if bc["raw_frame_hash"] != PARENT_RAW_HASH:
        raise ValueError("parent raw hash differs from preregistration")
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
    return entry, geometry, identity, episode_seed


def probe(selection_dir, capture_dir, mode):
    entry, geometry, identity, episode_seed = load_case(selection_dir, capture_dir)
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for parent-device comparison")
    config = entry["episode_config"]
    synth = SynthPointCloud(str(geometry), device="cuda:0")
    result = sample_pair(synth, config["start_pos"], config["start_yaw"],
                         episode_seed, identity, mode)
    result.update({
        "status": "RAW_FRAME_ORTHOGONAL_PROBE_COMPLETE",
        "mode": mode,
        "task_id": TASK_ID,
        "repeat": REPEAT,
        "matches_parent": [value == PARENT_RAW_HASH for value in result["raw_frame_hashes"]],
        "matches_stopped": [value == STOPPED_RAW_HASH for value in result["raw_frame_hashes"]],
        "parent_raw_frame_hash": PARENT_RAW_HASH,
        "stopped_raw_frame_hash": STOPPED_RAW_HASH,
        "episode_identity_hash": identity,
        "episode_rng_seed": episode_seed,
        "geometry_sha256": entry["geometry_sha256"],
        "selection_sha256": file_sha256(selection_dir / "selection.json"),
        "parent_trace_sha256": file_sha256(capture_dir / "paired_trace_transitions.jsonl"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "model_loaded": False,
        "policy_forward": False,
        "training_performed": False,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True,
                        choices=("plain", "preseed", "prebuild-rays"))
    args = parser.parse_args()
    try:
        result = probe(args.selection_dir, args.capture_dir, args.mode)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "RAW_FRAME_ORTHOGONAL_INPUT_ERROR", "detail": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
