"""Integrity and trajectory statistics, without training/GPU dependencies."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np


ROLES = ("anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON token: " + value)

    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    stack = [value]
    while stack:
        x = stack.pop()
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("nonfinite JSON number")
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return value


def read_json(path, max_bytes=64 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("JSON exceeds bound: " + str(path))
    return parse_json(path.read_text(encoding="utf-8"))


def _publish_generated(path, writer):
    """Publish one complete new file; a crash never creates a partial final name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("."+path.name+".partial_"+uuid.uuid4().hex)
    with partial.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Same-directory hard link is atomic and refuses an existing destination.
    # Interrupted partial files remain evidence; successful scratch is removed.
    os.link(partial, path)
    partial.unlink()


def write_json(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False)+"\n").encode("utf-8")
    _publish_generated(path, lambda stream: stream.write(content))


def write_npz(path, observation):
    _publish_generated(path, lambda stream:np.savez_compressed(stream,**observation))


def verify_checkpoint(path, expected):
    path = Path(path)
    names = {"header.json", "state.pt", "SHA256.json", "COMMITTED.json"}
    if path.is_symlink() or {p.name for p in path.iterdir() if p.is_file()} != names:
        raise ValueError("incomplete/ambiguous checkpoint: " + str(path))
    if any((path / n).is_symlink() for n in names):
        raise ValueError("checkpoint file symlink")
    if (path / "state.pt").stat().st_size > 64 * 1024 * 1024:
        raise ValueError("actor checkpoint exceeds 64 MiB bound")
    h = read_json(path / "header.json", 1024 * 1024)
    sums = read_json(path / "SHA256.json", 1024 * 1024)
    committed = read_json(path / "COMMITTED.json", 1024 * 1024)
    if set(sums) != {"header.json", "state.pt"}:
        raise ValueError("unexpected integrity manifest target")
    if (committed.get("schema_version") != 2 or committed.get("status") != "committed"
            or committed.get("files") != sums
            or committed.get("header_schema_hash") != sums["header.json"]):
        raise ValueError("invalid committed checkpoint")
    for name in sums:
        if file_hash(path / name) != sums[name]:
            raise ValueError("checkpoint bytes changed: " + name)
    if (h.get("schema_version") != 2 or h["update_index"] != expected["update"]
            or h["runtime_identity_hash"] != expected["runtime_identity_hash"]
            or canonical_hash(h["runtime_identity"]) != h["runtime_identity_hash"]
            or sums["header.json"] != expected["header_sha256"]
            or sums["state.pt"] != expected["state_sha256"]):
        raise ValueError("checkpoint differs from accepted training receipt")
    return h


def observation_hash(observation):
    h = hashlib.sha256()
    if set(observation) != {"agent_pos", "point_cloud"}:
        raise ValueError("unexpected observation keys")
    for key in sorted(observation):
        a = np.ascontiguousarray(observation[key], dtype=np.float32)
        if not np.isfinite(a).all():
            raise ValueError("nonfinite observation")
        h.update(key.encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(b"float32")
        h.update(a.tobytes())
    return h.hexdigest()


def action_views(full, policy, executed):
    arrays = [np.asarray(x, dtype=np.float32) for x in (full, policy, executed)]
    if [x.shape for x in arrays] != [(16, 11), (8, 11), (4, 11)]:
        raise ValueError("16/8/4 action shape violation")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("nonfinite action")
    if not (np.array_equal(arrays[0][2:10], arrays[1])
            and np.array_equal(arrays[0][2:6], arrays[2])):
        raise ValueError("action views disagree")
    return arrays


def trace_metrics(path, goal):
    p, g = np.asarray(path, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 1 or g.shape != (3,):
        raise ValueError("path/goal shape")
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("nonfinite executed path")
    segments = np.diff(p, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    distances = np.linalg.norm(p - g, axis=1)
    minimum = float(distances.min())
    for a, d in zip(p[:-1], segments):
        denom = float(d @ d)
        t = 0. if denom == 0. else float(np.clip(((g - a) @ d) / denom, 0., 1.))
        minimum = min(minimum, float(np.linalg.norm(a + t * d - g)))
    axis = g - p[0]
    norm = float(np.linalg.norm(axis))
    unit = axis / norm if norm else np.zeros(3)
    delta = p - p[0]
    cross = np.linalg.norm(delta - np.outer(delta @ unit, unit), axis=1)
    return {"path_length_m": float(lengths.sum()), "min_goal_distance_m": minimum,
            "final_goal_distance_m": float(distances[-1]),
            "backtrack_m": float(np.maximum(-(segments @ unit), 0.).sum()),
            "max_cross_track_m": float(cross.max()),
            "z_span_m": float(np.ptp(p[:, 2]))}


def paired_summary(records, treatment, reference):
    from .statistics import paired_summary as calculate
    return calculate(records, treatment, reference)
