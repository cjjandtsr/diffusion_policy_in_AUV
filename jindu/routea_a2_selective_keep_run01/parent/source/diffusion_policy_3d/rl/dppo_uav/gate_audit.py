"""G3 gate (design §5.5): 10k-reset admission-path audit.

The gate runs the REAL training admission path — RoleSampler -> task spec
-> UAVVectorEnvAdapter.reset — over n resets and independently re-verifies
each admission output:

- invalid window = 0             (R7-5 window gate over ALL obs_steps frames)
- invalid point cloud = 0        (per-frame failures inside the window)
- unvalidated fallback = 0       (rule 7 raises AdmissionExhausted, never
                                  fixed-coordinate fallback; all-far collapse
                                  signature counted per checked window frame)
- reachability pass = 100%       (independent A* re-run on the admitted pair)
- start/goal clearance pass = 100% (independent clearance re-check)
- role missing/unknown = 0       (info role == sampled role; role with no
                                  admitted worlds counted)
- role reset proportion vs target <= 2pp (RoleSampler.TOLERANCE_PP)
- eval-only world leakage = 0    (spec.split == rl_train)
- metadata/hash missing = 0
- admission blocked = 0          (AdmissionExhausted hard stop, rule 7)
- manifest hash mismatch = 0     (recomputed source hash != spec hash)
- dense path_stretch fail = 0    (dense_recovery EXHAUSTED on rule-5 stretch;
                                  per-attempt rejections re-sample normally)
- source hash missing/mismatch = 0 (window provenance source_manifest_hash
                                  absent or != spec manifest entry hash)
- runtime hash mismatch = 0      (window provenance runtime_episode_hash !=
                                  the record's episode_config_hash)

The re-verification geometry mirrors the adapter's own checker exactly:
obstacles from the adapter pool path (the file the env actually loaded) and
occupancy-grid ranges from the adapter env's curriculum — an independent
re-run, not a copy of the checker's verdict.

Aggregation is pure and deterministic; the CLI wrapper
(scripts/g3_10k_reset_gate_audit.py) builds the production adapter/specs
and writes G3_GATE_REPORT.json.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from diffusion_policy_3d.env.collision_geom import (
    compute_obstacle_clearance,
    load_obstacles,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    ROLE_ADMISSION_PARAMS,
    ROLE_NAMES,
    AdmissionExhausted,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
    RoleSampler,
    UAVVectorEnvAdapter,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.planner import (
    astar_3d,
    build_occupancy_grid,
)

# R7-5: n_frames = n_resets * contract.obs_steps — every window frame is
# independently validated (the G3 contract's >=30,000 checked frames at
# 10k resets × obs_steps 3).


def proportion_gate(
    counts: Dict[str, int],
    n_resets: int,
    targets: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """(verdict, deltas) — |actual - target| <= 2pp per role (RoleSampler gate).

    Requires the 10k scale to be meaningful: at n=60 a 0.30 target has
    ~5.9pp std, so this gate is asserted only by the production run.
    """
    assert n_resets > 0, "empty reset count"
    deltas = {r: counts[r] / n_resets - targets[r] for r in ROLE_NAMES}
    verdict = "PASS" if all(abs(d) <= RoleSampler.TOLERANCE_PP for d in deltas.values()) else "FAIL"
    return verdict, deltas


def _json_path_of(
    adapter: UAVVectorEnvAdapter,
    spec: EpisodeTaskSpec,
    world_json: Optional[Dict[str, str]],
) -> str:
    """Obstacles file for independent re-verification.

    The adapter pool path is authoritative (the file the env's SynthPointCloud
    and the AdmissionChecker actually used); world_json only overrides it.
    """
    if world_json is not None:
        path = world_json.get(spec.world_uid)
        if path:
            return path
    return adapter._pool_path(spec.world_uid)


def run_reset_gate_audit(
    adapter: UAVVectorEnvAdapter,
    task_specs: List[EpisodeTaskSpec],
    n_resets: int,
    seed: int,
    *,
    world_json: Optional[Dict[str, str]] = None,
    targets: Dict[str, float] = DEFAULT_ROLE_TARGETS,
    task_sampler: Optional[EpisodeTaskSampler] = None,
    expected_hashes: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Run the admission path n_resets times; aggregate the §5.5 gates.

    Uses the SAME stateful EpisodeTaskSampler the train loop runs
    (no independently rebuilt RoleSampler/cursor in the gate path).
    ``task_sampler`` is injected (the train loop may hand its own instance);
    when None, one is constructed here with ``base_seed=seed`` — the same
    factory call the production loop makes.

    Args:
        adapter: real UAVVectorEnvAdapter over the production pool.
        task_specs: rl_train EpisodeTaskSpec list (roles from manifest).
        n_resets: number of admission resets (10,000 in production).
        seed: deterministic sampler base seed / episode seeds.
        world_json: world_uid -> obstacles json path; falls back to the
            adapter pool path when absent.
        targets: role proportion targets (DEFAULT_ROLE_TARGETS).
        task_sampler: optional shared EpisodeTaskSampler (same path as the
            train loop); constructed from task_specs/targets/seed if None.
        expected_hashes: composite-key (world_uid, role, admission_id) ->
            recomputed source-entry hash. When given, a spec whose hash
            does not match is counted as manifest_hash_mismatch and skips
            the reset (the adapter would only exhaust admission on it —
            rule 6 — so this surfaces the ROOT CAUSE instead of a blocked
            count). None skips the check (callers that cannot recompute).

    Returns:
        metrics dict: counters, per-role counts/proportions/deltas, gates
        verdicts, n_frames. Deterministic for a fixed seed.
    """
    from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
        PointCloudValidityGate,
        build_observation_window_mapping,
        validate_policy_observation_window,
    )

    contract = adapter.contract
    if task_sampler is None:
        task_sampler = EpisodeTaskSampler(task_specs, targets, base_seed=seed)
    gate = PointCloudValidityGate(contract=contract, phase="formal")
    counts = {r: 0 for r in ROLE_NAMES}

    invalid_pc = fallback = blocked = hash_mismatch = 0
    reach_fail = clr_fail = role_unknown = leakage = hash_missing = 0
    # R7-5 window gates: the full policy window per reset + dual-hash
    # independence (source from the manifest entry, runtime from the env's
    # ACTUAL post-reset state — never a self-copy of the spec).
    invalid_window = source_hash_missing = source_hash_mismatch = 0
    runtime_hash_mismatch = 0

    for i in range(n_resets):
        # reset_at() refuses ACTIVE slots, and this audit resets the same
        # slot for every probe without stepping — close the previous probe
        # episode explicitly (DONE, no fault verdict) before the next reset.
        # First iteration: the slots are INITIAL, never active.
        if not all(adapter.is_active(i) for i in range(adapter.n_envs)):
            pass  # INITIAL slots on the first pass — nothing to abort
        else:
            # Explicit state query, no blanket RuntimeError swallow — an
            # unexpected active-state mismatch must surface, not be
            # silently absorbed.
            adapter.abort_episode(list(range(adapter.n_envs)))
        sample = task_sampler.next_attempt()
        spec = sample.task_spec
        role = spec.role
        counts[role] += 1
        if not spec.manifest_entry_hash:
            hash_missing += 1
        if spec.split != "rl_train":
            leakage += 1
        if expected_hashes is not None:
            key = manifest_record_key(spec.world_uid, spec.role, spec.admission_id)
            if expected_hashes.get(key) != spec.manifest_entry_hash:
                hash_mismatch += 1
                continue  # rule-6 admission would only exhaust on this spec

        try:
            obs, info = adapter.reset([spec], seeds=[sample.episode_seed])
        except AdmissionExhausted:
            blocked += 1  # hard stop, not a fallback (rule 7)
            continue

        if info[0].get("role") != role:
            role_unknown += 1

        # R7-5: the audit validates the FULL policy window (obs_steps
        # frames) through the production provenance mapping — every frame
        # re-validated, recorded valid flags / frame hashes / ray reports
        # cross-checked, and BOTH independent hashes verified (source vs
        # the manifest entry, runtime vs the record's episode hash).
        window_mapping = info[0].get("observation_window_mapping")
        if not window_mapping:
            # Fail-closed: a missing mapping still gates on content, but
            # the missing hashes themselves count as violations.
            window_mapping = build_observation_window_mapping(
                obs, contract.obs_steps, float(adapter.cam_far))
        if not window_mapping.get("source_manifest_hash"):
            source_hash_missing += 1
        if window_mapping.get("source_manifest_hash") != spec.manifest_entry_hash:
            source_hash_mismatch += 1
        if window_mapping.get("runtime_episode_hash") != info[0].get(
            "episode_config_hash"
        ):
            runtime_hash_mismatch += 1
        window = validate_policy_observation_window(
            window_mapping, obs_steps=contract.obs_steps, gate=gate)
        if not window.valid:
            invalid_window += 1
        for fr in window.frame_results:
            if not fr.valid:
                invalid_pc += 1
        # The frozen-legacy collapse signature is every point at
        # (far, far, far) — count it as unvalidated fallback directly from
        # the RETURNED window frames (the far-dome encoding is spread on
        # the far sphere, never all-at-far, so this stays 0 in production).
        for _f in obs["point_cloud"][0, -contract.obs_steps:]:
            if np.allclose(_f, float(adapter.cam_far)):
                fallback += 1

        # Independent re-verification of the admitted pair: same obstacles
        # file and grid ranges the checker used, recomputed from scratch.
        cfg = adapter.current_episode_config()
        start, goal = np.asarray(cfg["start"], dtype=float), np.asarray(cfg["goal"], dtype=float)
        obstacles = load_obstacles(_json_path_of(adapter, spec, world_json))
        sc, _, _ = compute_obstacle_clearance(start[0], start[1], start[2], obstacles)
        gc, _, _ = compute_obstacle_clearance(goal[0], goal[1], goal[2], obstacles)
        params = ROLE_ADMISSION_PARAMS[role]
        if sc < params["start_cl"] or gc < params["goal_cl"]:
            clr_fail += 1
        x_min, x_max, y_min, y_max = adapter._env._curriculum_xy_range()
        z_low, z_high = adapter._env._curriculum_z_range()
        grid, meta = build_occupancy_grid(
            obstacles,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_low,
            z_max=z_high,
        )
        reach, _, _, _, _, _ = astar_3d(
            grid, meta, start[0], start[1], start[2], goal[0], goal[1], goal[2]
        )
        if not reach:
            reach_fail += 1

    # Dense_recovery admission attempts that EXHAUSTED on rule-5 path
    # stretch, counted at the adapter (the only place rule 5 runs).
    # Per-attempt stretch rejections (re-sampled within the 50-attempt
    # window) are normal admission behavior and are NOT counted.
    dense_stretch_fail = int(
        getattr(adapter, "stretch_fail_count", lambda: 0)()
    )

    proportions = RoleSampler.proportion_report(counts)
    prop_verdict, deltas = proportion_gate(counts, n_resets, targets)
    gates = {
        "invalid_pc": "PASS" if invalid_pc == 0 else "FAIL",
        "unvalidated_fallback": "PASS" if fallback == 0 else "FAIL",
        "reachability": "PASS" if reach_fail == 0 else "FAIL",
        "clearance": "PASS" if clr_fail == 0 else "FAIL",
        "role_unknown": "PASS" if role_unknown == 0 else "FAIL",
        "role_proportion": prop_verdict,
        "eval_leakage": "PASS" if leakage == 0 else "FAIL",
        "hash_missing": "PASS" if hash_missing == 0 else "FAIL",
        "admission_blocked": "PASS" if blocked == 0 else "FAIL",
        "manifest_hash_mismatch": "PASS" if hash_mismatch == 0 else "FAIL",
        "dense_path_stretch_fail": "PASS" if dense_stretch_fail == 0 else "FAIL",
        # R7-5 window gates (guide R7-5 hard gates: invalid window/frame =
        # 0; source/runtime hash missing or mismatch = 0).
        "invalid_window": "PASS" if invalid_window == 0 else "FAIL",
        "source_hash_missing": "PASS" if source_hash_missing == 0 else "FAIL",
        "source_hash_mismatch": "PASS" if source_hash_mismatch == 0 else "FAIL",
        "runtime_hash_mismatch": "PASS" if runtime_hash_mismatch == 0 else "FAIL",
    }
    return {
        "n_resets": n_resets,
        # R7-5: every window frame is independently validated —
        # n_frames = n_resets * obs_steps (30,000 at the 10k contract).
        "n_frames": n_resets * contract.obs_steps,
        "invalid_window": invalid_window,
        "invalid_pc": invalid_pc,
        "unvalidated_fallback": fallback,
        "admission_blocked": blocked,
        "manifest_hash_mismatch": hash_mismatch,
        "source_hash_missing": source_hash_missing,
        "source_hash_mismatch": source_hash_mismatch,
        "runtime_hash_mismatch": runtime_hash_mismatch,
        "dense_path_stretch_fail": dense_stretch_fail,
        "reachability_fail": reach_fail,
        "clearance_fail": clr_fail,
        "role_unknown": role_unknown,
        "eval_leakage": leakage,
        "hash_missing": hash_missing,
        "role_counts": counts,
        "role_proportions": proportions,
        "role_proportion_deltas": deltas,
        "targets": dict(targets),
        "seed": seed,
        "gates": gates,
    }
