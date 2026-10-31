#!/usr/bin/env python3
"""Offline v17 preparation; no policy loading, rollout or training.

``density`` consumes a frozen-demo export JSON with keys ``episodes`` and
``source_evidence``. Each episode has episode_id/source/split/start/goal/
ground_z/initial_yaw/positions, as specified by build_density. The source
evidence lists ``files`` with path+sha256 and ``coordinate_reference`` with
the explicit derivation of start/goal/ground/yaw. Missing references are an
input deficiency, never an excuse to manufacture zero ground or density.

``calibrate`` consumes a preregistered JSON containing ``episodes`` (complete
executed trajectories), ``comparisons`` (better/worse episode IDs), and
``source_evidence`` (same file identity schema). Episode fields are documented
in reward_v17_support.calibration.replay_episodes. ``ordering_categories``
maps each required category below to its registered comparison pairs.

Only a resolved result writes a new formal config. An unresolved result writes
its diagnostic report and exits 2. Existing files are never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.env.reward_v17 import resolve_config
from diffusion_policy_3d.env.reward_v17_support.config import DEFAULTS
from diffusion_policy_3d.env.reward_v17_support.density import build_density, content_hash, validate_density
from diffusion_policy_3d.env.reward_v17_support.geometry import GeometryContext, path_geometry
from diffusion_policy_3d.env.reward_v17_support.calibration import (
    anchor_scale, replay_episodes, select_height_weights, deterministic_signal_checks, _source,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v17.facade import RewardV17Facade

ORDERINGS = ("necessary_overflight_vs_collision", "necessary_overflight_vs_ground",
             "necessary_overflight_vs_timeout", "safe_vs_roof_5cm",
             "side_route_vs_failure", "overflight_vs_failure",
             "safe_vs_redundant_high", "high_goal_direct_vs_support_detour",
             "safe_vs_clear_loop", "safe_vs_repeated_trap")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    # These are generated WSL runtime artifacts, not source-file edits.
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _inside(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def evidence(value, relative_to, *, repo_root=ROOT):
    source = value.get("source_evidence", {})
    if not source.get("coordinate_reference") or not source.get("files"):
        raise ValueError("missing original source files or explicit coordinate-reference evidence")
    entries = source["files"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 10000:
        raise ValueError("source evidence files must be a bounded list")
    path_base = source.get("path_base", "artifact_parent")
    if path_base not in ("artifact_parent", "repo_root"):
        raise ValueError("unsupported source evidence path_base")
    base = Path(repo_root if path_base == "repo_root" else relative_to).resolve()
    verified = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("source evidence file schema mismatch")
        logical = str(entry["path"]).replace("\\", "/")
        digest = str(entry["sha256"]).lower()
        if not logical or logical in seen:
            raise ValueError("source evidence paths must be unique and non-empty")
        if not _SHA256.fullmatch(digest):
            raise ValueError("source evidence sha256 is invalid")
        seen.add(logical)
        candidate = Path(entry["path"])
        unresolved = candidate if candidate.is_absolute() else base / candidate
        if unresolved.is_symlink():
            raise ValueError("source evidence must be a regular non-symlink file")
        path = unresolved.resolve()
        if path_base == "repo_root" and not _inside(path, base):
            raise ValueError("source evidence path escapes repo_root")
        if not path.is_file() or sha256(path) != entry["sha256"].lower():
            raise ValueError("source evidence missing/changed: {}".format(path))
        verified.append(dict(path=logical, sha256=digest))
    return dict(files=verified, coordinate_reference=source["coordinate_reference"],
                path_base=path_base,
                evidence_scope="verified_file_bytes_and_explicit_export_references")


def build_frozen_density(args):
    path = Path(args.input).resolve()
    value = read_json(path)
    source = evidence(value, path.parent)
    # Identity names the immutable exported dataset actually consumed, not an
    # unverified claim that its digest equals an entire original Zarr tree.
    density = build_density(value["episodes"], dataset_sha256=sha256(path))
    density.pop("content_hash")
    density.update(dataset_identity_scope="frozen_demonstration_export_json_bytes",
                   source_evidence=source, source_export_path=str(path))
    density["content_hash"] = content_hash(density)
    validate_density(density)
    write_json(args.output, density)
    print(json.dumps(dict(status="written_not_training_calibrated", output=str(args.output),
                          sha256=sha256(args.output), episodes=density["episode_count"],
                          frames=density["sample_count"], demo_height_max=density["demo_height_max"])))
    return 0


def measure_anchors(episodes, geometry_cfg):
    """Geometry-only pass. No provisional d_ref/s_f or trial reward is used."""
    empirical, fixtures = [], []
    for episode in episodes:
        fixture = episode.get("split") == "fixture"
        _source(episode, fixture)
        if episode["expected_done_reason"] != "reached":
            continue
        context = GeometryContext(**episode["geometry_context"])
        obstacles = [ObstacleSpec.from_dict(o) for o in episode["obstacles"]]
        goal = np.asarray(episode["goal"], dtype=float)
        previous = np.asarray(episode["start"], dtype=float)
        local = []
        for index, step in enumerate(episode["steps"]):
            path = np.asarray(step["executed_path"], dtype=float)
            if path.shape != (5, 3) or not np.array_equal(path[0], previous):
                raise ValueError("anchor path is not a complete continuous H4 sequence")
            g = path_geometry(path, obstacles, context, geometry_cfg)
            reached = np.linalg.norm(path[-1] - goal) < .5 and abs(path[-1, 2] - goal[2]) < .3
            if not g["physical_safe"] or (reached != (index == len(episode["steps"]) - 1)):
                raise ValueError("anchor claimed safe reached but current physical truth disagrees")
            if g["related_count"] and 0 < g["d_soft"] < 100:
                local.append(dict(episode_id=episode["episode_id"], world_id=episode["world_id"],
                                  role=episode["role"], split=episode["split"], d_soft=g["d_soft"]))
            previous = path[-1]
        (fixtures if fixture else empirical).extend(local)
    return empirical, fixtures


def check_ordering_manifest(manifest):
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    if len(episodes) != len(manifest["episodes"]):
        raise ValueError("duplicate episode IDs")
    categories = manifest.get("ordering_categories", {})
    missing = [key for key in ORDERINGS if not categories.get(key)]
    if missing:
        raise ValueError("preregistered full-return comparisons missing: {}".format(missing))
    pairs = []
    for category in ORDERINGS:
        for better, worse in categories[category]:
            a, b = episodes[better], episodes[worse]
            if a["split"] != "fixture" or b["split"] != "fixture":
                raise ValueError("height-weight ordering set must be preregistered geometric fixtures")
            for key in ("start", "goal", "obstacles", "geometry_context"):
                if a[key] != b[key]:
                    raise ValueError("ordering pair changes the task: {}/{}".format(better, worse))
            if a["expected_done_reason"] != "reached":
                raise ValueError("reference ordering path must safely reach")
            if category in ("safe_vs_roof_5cm", "safe_vs_redundant_high", "high_goal_direct_vs_support_detour") and b["expected_done_reason"] != "reached":
                raise ValueError("efficiency/clearance ordering must compare two successful paths")
            expected_failure = {"necessary_overflight_vs_collision": "collision", "necessary_overflight_vs_ground": "collision",
                                "necessary_overflight_vs_timeout": "time_limit"}.get(category)
            if expected_failure is not None and b["expected_done_reason"] != expected_failure:
                raise ValueError("registered failure category does not match its terminal semantics")
            if category.startswith("necessary_overflight"):
                # Explicit necessity witness from design section 5.3: a
                # ground-sitting wall spans the entire legal Y domain.
                context = GeometryContext(**a["geometry_context"])
                witness = False
                for raw in a["obstacles"]:
                    o = ObstacleSpec.from_dict(raw)
                    if o.shape != "box" or o.yaw != 0.:
                        continue
                    spanning = o.y - o.size_y / 2 <= context.xy_min[1] and o.y + o.size_y / 2 >= context.xy_max[1]
                    grounded = o.z - o.size_z / 2 <= context.ground_z
                    low, high = o.x - o.size_x / 2 - .1, o.x + o.size_x / 2 + .1
                    opposite = min(a["start"][0], a["goal"][0]) < low and max(a["start"][0], a["goal"][0]) > high
                    witness |= spanning and grounded and opposite
                if not witness:
                    raise ValueError("necessary-overflight fixture has no full-domain grounded-wall witness")
            if (better, worse) not in pairs:
                pairs.append((better, worse))
    return pairs


def replay_contract_checks(manifest, records, density):
    """Check the claimed counterexample features, not just category names."""
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    checks = {}
    for category, pairs in manifest["ordering_categories"].items():
        for better, worse in pairs:
            a, b = records[better], records[worse]
            am, bm = [r["metrics"] for r in a], [r["metrics"] for r in b]
            if category == "necessary_overflight_vs_ground":
                check = bm[-1]["ground_clearance"] < .2
            elif category == "safe_vs_roof_5cm":
                clearances = [m["overflight_clearance_min"] for m in bm if m["overflight_clearance_min"] is not None]
                check = bool(clearances) and abs(min(clearances) - .05) < 1e-6
            elif category in ("safe_vs_redundant_high", "high_goal_direct_vs_support_detour"):
                check = (max(m["max_z"] for m in bm) > max(m["max_z"] for m in am)
                         and sum(m["positive_climb_m"] for m in bm) > sum(m["positive_climb_m"] for m in am))
                if category == "high_goal_direct_vs_support_detour":
                    e = episodes[better]
                    check = (check and e["goal"][2] - e["geometry_context"]["ground_z"] > density["demo_height_max"]
                             and bm[-1]["support_eligible_above_demo"] > am[-1]["support_eligible_above_demo"])
            elif category == "safe_vs_clear_loop":
                check = any(m["loop_detected"] for m in bm)
            elif category == "safe_vs_repeated_trap":
                check = max(m["trap_segment_id"] for m in bm) >= 2
            else:
                check = True  # physical outcome/task identity checked during replay
            checks[category + ":" + better + ">" + worse] = bool(check)
    guard_rows = [row["metrics"] for e in manifest["episodes"] if e["split"] == "fixture"
                  for row in records[e["episode_id"]] if row["metrics"]["guard_detour_opportunity"] or row["metrics"]["guard_loop_window_qualified"]]
    checks["fixture_guard_cap_fraction"] = bool(guard_rows) and sum(m["guard_cap_hit"] for m in guard_rows) / len(guard_rows) <= .05
    return checks


def calibrate(args):
    manifest_path, density_path = Path(args.input).resolve(), Path(args.density).resolve()
    manifest, density = read_json(manifest_path), validate_density(read_json(density_path))
    source = evidence(manifest, manifest_path.parent)
    pairs = check_ordering_manifest(manifest)
    base = OmegaConf.load(args.base_config)
    if int(base.run.updates) != 256 or int(base.environment.max_episode_steps) != 60:
        raise ValueError("formal base must retain the approved 256-update/60-step protocol")
    if float(base.dppo.gamma_environment) != .99:
        raise ValueError("this finite calibration protocol binds environment gamma=.99")
    # Actual factory currently exposes disturbance enable, retaining scales
    # exec_noise=0 and conditional disturb_pos=0.3. Selector checks the same.
    sigma = .3 if bool(base.environment.disturb_enable) else 0.
    empirical, fixtures = measure_anchors(manifest["episodes"], DEFAULTS)
    scale = anchor_scale(empirical, fixtures)
    report = dict(status="unresolved", reward_schema_version="v17", design_revision="v1.4",
                  density_hash=density["content_hash"], anchor=scale, source_evidence=source,
                  manifest_sha256=sha256(manifest_path), base_config_sha256=sha256(args.base_config),
                  reward_source_sha256=RewardV17Facade().identity.source_sha256,
                  policy_updated=False, fixture_evidence_not_policy_performance=True)
    if scale["status"] == "resolved":
        cfg = resolve_config(dict(gamma_pbrs=.99, d_ref=scale["d_ref"], s_f=scale["s_f"],
                                  w_low=2.5, w_high=1., w_climb=1., sigma_z=sigma))
        features, _, records = replay_episodes(manifest["episodes"], cfg, density)
        checks, signals = deterministic_signal_checks(cfg, density, with_evidence=True)
        checks.update(replay_contract_checks(manifest, records, density))
        fixture_features = {e["episode_id"]: features[e["episode_id"]] for e in manifest["episodes"] if e["split"] == "fixture"}
        search = select_height_weights(fixture_features, pairs, checks)
        report.update(search=search, features=features, replay_records=records, signal_precision_evidence=signals,
                      replay_records_reference_weights={key: cfg[key] for key in ("w_low", "w_high", "w_climb")},
                      feature_cache_scope="one_full_reward_replay_then_27_scalar_aggregations")
        if search["status"] == "resolved":
            selected = search["selected"]
            cfg.update({key: selected[key] for key in ("w_low", "w_high", "w_climb")})
            cfg = resolve_config(cfg)
            discount_sum = sum(cfg["gamma_pbrs"] ** t for t in range(cfg["max_episode_steps"]))
            bound = cfg["w_low"] + cfg["w_high"] + cfg["H_e"] * cfg["w_climb"]
            report.update(status="resolved", selected_config=cfg,
                          vertical_bound_discounted=discount_sum * bound,
                          vertical_bound_undiscounted=cfg["max_episode_steps"] * bound)
    report_path = Path(args.output).resolve()
    write_json(report_path, report)
    if report["status"] == "resolved":
        config_path = Path(args.formal_config).resolve()
        base.reward_version = "v17"
        base.reward = report["selected_config"]
        base.reward_density = dict(path=str(density_path), sha256=sha256(density_path))
        base.reward_calibration = dict(path=str(report_path), sha256=sha256(report_path))
        base.output_dir = args.training_output
        with config_path.open("x", encoding="utf-8") as stream:
            stream.write(OmegaConf.to_yaml(base, resolve=True))
    print(json.dumps(dict(status=report["status"], report=str(report_path),
                          formal_config=str(args.formal_config) if report["status"] == "resolved" else None)))
    return 0 if report["status"] == "resolved" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="operation", required=True)
    density = sub.add_parser("density", help="re-encode verified frozen demonstration export")
    density.add_argument("--input", required=True)
    density.add_argument("--output", required=True)
    density.set_defaults(func=build_frozen_density)
    calibration = sub.add_parser("calibrate", help="one offline replay; never loads a policy")
    calibration.add_argument("--input", required=True)
    calibration.add_argument("--density", required=True)
    calibration.add_argument("--base-config", default=str(ROOT / "diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_formal_seed1_contactfix_v1.yaml"))
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--formal-config", required=True)
    calibration.add_argument("--training-output", required=True)
    calibration.set_defaults(func=calibrate)
    args = parser.parse_args()
    for key in ("output", "formal_config"):
        value = getattr(args, key, None)
        if value and Path(value).exists():
            parser.error("refusing to overwrite existing artifact: {}".format(value))
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps(dict(status="input_or_semantic_error", detail=str(exc)), ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
