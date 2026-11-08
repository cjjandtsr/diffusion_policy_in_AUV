"""Frozen demonstration probabilities in route-relative, ground-aware cells."""

import hashlib
import json
import math
from collections import Counter

import numpy as np

COORDINATE_VERSION = "route_xy_goal_origin_ground_z_ties_even_v1"
BINS = (1., .5, .5)


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def support_cell(position, start, goal, ground_z, initial_yaw):
    p, s, g = (np.asarray(x, dtype=np.float64) for x in (position, start, goal))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (p, s, g)) or not np.isfinite([ground_z, initial_yaw]).all():
        raise ValueError("density coordinates must be finite xyz with explicit ground/yaw")
    axis = g[:2] - s[:2]
    norm = np.hypot(*axis)
    axis = axis / norm if norm > 0 else np.array([math.cos(initial_yaw), math.sin(initial_yaw)])
    offset = p[:2] - g[:2]
    values = (np.dot(offset, axis), np.dot(offset, [-axis[1], axis[0]]), p[2] - ground_z)
    return tuple(int(round(float(x) / b)) for x, b in zip(values, BINS))


def build_density(episodes, *, dataset_sha256):
    if len(dataset_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in dataset_sha256):
        raise ValueError("density requires actual source dataset SHA256")
    counts, episode_ids, references, height_max = Counter(), set(), [], -math.inf
    for e in episodes:
        if e.get("source") not in ("frozen_bc", "frozen_ego") or e.get("split") != "bc_demo":
            raise ValueError("density accepts frozen demonstrations only, not RL/evaluation/fixtures")
        eid = str(e["episode_id"])
        if eid in episode_ids:
            raise ValueError("duplicate demonstration episode")
        episode_ids.add(eid)
        positions = np.asarray(e["positions"], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError("density episode positions must be nonempty Nx3")
        for p in positions:
            cell = support_cell(p, e["start"], e["goal"], e["ground_z"], e["initial_yaw"])
            counts[",".join(map(str, cell))] += 1
        height_max = max(height_max, float(np.max(positions[:, 2] - e["ground_z"])))
        references.append({key: e[key] for key in ("episode_id", "start", "goal", "ground_z", "initial_yaw")})
    if not counts:
        raise ValueError("empty density source")
    total = sum(counts.values())
    probs = sorted(n / total for n in counts.values())
    result = dict(coordinate_version=COORDINATE_VERSION, bins=list(BINS),
                  dataset_sha256=dataset_sha256, source="frozen_demonstrations",
                  episode_count=len(episode_ids), sample_count=total, counts=dict(counts),
                  episode_references=references, demo_height_max=height_max,
                  tau_D=probs[max(0, math.ceil(.2 * len(probs)) - 1)])
    result["content_hash"] = content_hash(result)
    return result


def validate_density(density):
    if not isinstance(density, dict) or density.get("coordinate_version") != COORDINATE_VERSION:
        raise ValueError("v17 requires rebuilt frozen density with explicit coordinate version")
    if density.get("source") != "frozen_demonstrations" or density.get("bins") != list(BINS):
        raise ValueError("density source/bin mismatch")
    if not math.isfinite(density.get("demo_height_max", float("nan"))):
        raise ValueError("density requires measured maximum ground-relative demonstration height")
    if len(density.get("episode_references", [])) != density.get("episode_count"):
        raise ValueError("density missing per-episode coordinate provenance")
    payload = {k: v for k, v in density.items() if k != "content_hash"}
    if density.get("content_hash") != content_hash(payload):
        raise ValueError("density content hash mismatch")
    counts = density.get("counts", {})
    if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("density counts must be positive integers")
    if sum(counts.values()) != density.get("sample_count"):
        raise ValueError("density sample count mismatch")
    probs = sorted(n / density["sample_count"] for n in counts.values())
    if density.get("tau_D") != probs[max(0, math.ceil(.2 * len(probs)) - 1)]:
        raise ValueError("density probability threshold mismatch")
    return density


def probability(density, cell):
    return density["counts"].get(",".join(map(str, cell)), 0) / density["sample_count"]
