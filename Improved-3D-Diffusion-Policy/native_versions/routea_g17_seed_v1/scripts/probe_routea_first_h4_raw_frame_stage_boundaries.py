"""Read-only stage boundaries for the first-H4 initial raw-frame mismatch."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.model.common import synth_pointcloud as synth_module
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
    canonical_array_hash,
)
from scripts.probe_routea_first_h4_raw_frame_orthogonal import (
    PARENT_RAW_HASH,
    REPEAT,
    STOPPED_RAW_HASH,
    TASK_ID,
    load_case,
)
from scripts.routea_causal_diagnostics.reporting import file_sha256


STAGE_ORDER = (
    "ray_dir_world", "t_cylinder", "t_box", "t_ground",
    "pre_noise", "post_noise", "raw_frame",
)


class ReproductionMismatch(ValueError):
    """The instrumented pair did not reproduce the already-observed pair."""


def _copy_float32(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError("stage tensor must be float32")
    return np.ascontiguousarray(array).copy()


def compare_stage_pair(first, second):
    """Compare the same ordered stage boundaries without treating inf misses as NaN."""
    if set(first) != set(second):
        raise ValueError("stage sets differ between calls")
    if not {"ray_dir_world", "t_ground", "pre_noise", "post_noise", "raw_frame"} <= set(first):
        raise ValueError("required stage boundary is absent")
    summaries = {}
    first_mismatch = None
    for name in STAGE_ORDER:
        if name not in first:
            continue
        a, b = _copy_float32(first[name]), _copy_float32(second[name])
        if a.shape != b.shape:
            raise ValueError("stage shapes differ: " + name)
        different = a.view(np.uint32) != b.view(np.uint32)
        count = int(np.count_nonzero(different))
        finite_both = np.isfinite(a) & np.isfinite(b)
        max_abs = (float(np.max(np.abs(a[finite_both].astype(np.float64)
                                       - b[finite_both].astype(np.float64))))
                   if np.any(finite_both) else None)
        summaries[name] = {
            "first_hash": canonical_array_hash(a),
            "second_hash": canonical_array_hash(b),
            "shape": list(a.shape),
            "bitwise_exact": count == 0,
            "different_elements": count,
            "first_index": None if count == 0 else np.argwhere(different)[0].tolist(),
            "max_abs_finite": max_abs,
        }
        if count and first_mismatch is None:
            first_mismatch = name
    return {"stage_order": list(summaries),
            "first_mismatch_stage": first_mismatch,
            "stages": summaries}


def trace_two_samples(synth, start_pos, start_yaw, episode_seed, identity):
    """Observe two unchanged synthesis calls; restore all temporary wrappers."""
    position = np.asarray(start_pos, dtype=np.float32).copy()
    position[2] = max(position[2], 0.05)
    snapshots = []
    hashes = []
    hit_counts = []
    ray_counts = []
    current = None

    def record(name, value):
        if name not in current:
            current[name] = _copy_float32(value)

    def wrap_intersection(name, original):
        def traced(ray_origin, ray_dir, *args):
            record("ray_dir_world", ray_dir)
            output = original(ray_origin, ray_dir, *args)
            record(name, output)
            return output
        return traced

    original_noise = synth._apply_noise

    def traced_noise(pc, *args, **kwargs):
        record("pre_noise", pc)
        output = original_noise(pc, *args, **kwargs)
        record("post_noise", output)
        return output

    with patch.object(synth_module, "intersect_rays_cylinder", wrap_intersection(
            "t_cylinder", synth_module.intersect_rays_cylinder)), \
         patch.object(synth_module, "intersect_rays_box", wrap_intersection(
             "t_box", synth_module.intersect_rays_box)), \
         patch.object(synth_module, "intersect_rays_ground", wrap_intersection(
             "t_ground", synth_module.intersect_rays_ground)), \
         patch.object(synth, "_apply_noise", traced_noise):
        for _ in range(2):
            current = {}
            frame, meta = synth.generate_with_metadata(
                position, float(start_yaw), num_points=4096, add_noise=True,
                rng_seed=int(episode_seed), episode_identity_hash=identity,
                decision_index=0,
            )
            record("raw_frame", frame)
            if canonical_array_hash(frame) != meta.raw_array_hash:
                raise ValueError("metadata raw-frame hash differs from frame")
            snapshots.append(current)
            hashes.append(meta.raw_array_hash)
            hit_counts.append(int(meta.valid_hit_rays))
            ray_counts.append(int(meta.total_rays))

    result = compare_stage_pair(snapshots[0], snapshots[1])
    result.update({"raw_frame_hashes": hashes,
                   "valid_hit_rays": hit_counts,
                   "total_rays": ray_counts})
    return result


def require_reproduced_pair(hashes):
    if list(hashes) != [STOPPED_RAW_HASH, PARENT_RAW_HASH]:
        raise ReproductionMismatch("expected raw-frame pair not reproduced: " +
                                   json.dumps(list(hashes)))


def probe(selection_dir, capture_dir):
    entry, geometry, identity, episode_seed = load_case(selection_dir, capture_dir)
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for parent-device comparison")
    config = entry["episode_config"]
    synth = SynthPointCloud(str(geometry), device="cuda:0")
    result = trace_two_samples(synth, config["start_pos"], config["start_yaw"],
                               episode_seed, identity)
    require_reproduced_pair(result["raw_frame_hashes"])
    result.update({
        "status": "RAW_FRAME_STAGE_BOUNDARY_PROBE_COMPLETE",
        "task_id": TASK_ID,
        "repeat": REPEAT,
        "episode_identity_hash": identity,
        "episode_rng_seed": episode_seed,
        "geometry_sha256": entry["geometry_sha256"],
        "selection_sha256": file_sha256(selection_dir / "selection.json"),
        "parent_trace_sha256": file_sha256(capture_dir / "paired_trace_transitions.jsonl"),
        "stopped_raw_frame_hash": STOPPED_RAW_HASH,
        "parent_raw_frame_hash": PARENT_RAW_HASH,
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
    args = parser.parse_args()
    try:
        result = probe(args.selection_dir, args.capture_dir)
    except ReproductionMismatch as exc:
        print(json.dumps({"status": "RAW_FRAME_STAGE_BOUNDARY_REPRODUCTION_FAILED",
                          "detail": str(exc)}, sort_keys=True), file=sys.stderr)
        return 10
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "RAW_FRAME_STAGE_BOUNDARY_INPUT_ERROR",
                          "detail": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
