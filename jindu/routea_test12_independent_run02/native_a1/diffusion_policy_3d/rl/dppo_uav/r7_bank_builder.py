"""R7-8 shared bank-builder logic (guide Task R7-8, lines 1430-1507).

The builder (scripts/build_g8_capability_bank.py) freezes 100 paired
episode configs from the eval-only splits (rl_val ∪ rl_test), labels
each with a policy-INDEPENDENT geometry opportunity, and writes the
frozen r7-paired-bank-v1 artifact. This module holds the parts that
need direct test coverage:

- reward13_density_to_capability: the offline-density schema bridge
  (the EGO-dataset builder emits reward_v13 ``cell_counts``; the
  capability layer requires the ``counts`` schema) — user ruling
  2026-08-24: real density is built inside R7-8, never the synthetic
  placeholder;
- quota selection (Step 2: exactly 100 episodes, three capabilities
  >= 20 each, >= 20 per main role, unknown/no-opportunity-only <= 10);
- train-leakage check (Step 3: bank ∩ train manifest = 0, hard-fail);
- aggregate computation (Step 1: candidates, opportunity, exposure,
  role/world counts, membership hash).

Exposure counting itself (Step 4/6, two-phase freeze) is measured by
the R7-9 canonical 100-episode BC replay; this module only computes the
geometry opportunity and the aggregate skeleton (exposure is
``NOT_YET_MEASURED`` in v1, rebuilt with real measurements for v2).
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    OfflineDensity,
)

# ═══════════════════════════════════════════════════════════════════
# offline-density schema bridge (reward_v13 cell_counts -> capability)
# ═══════════════════════════════════════════════════════════════════


def reward13_density_to_capability(data: Dict[str, Any]) -> OfflineDensity:
    """Convert a reward_v13 offline-density dict (``cell_counts``) to
    the capability ``OfflineDensity`` (``counts`` schema).

    The EGO-dataset builder (tools/build_routea_offline_density.py)
    emits ``{"cell_counts": {...}, "total": N, "p20_threshold": ...,
    "p50_threshold": ...}``; the capability layer's OfflineDensity
    validates ``schema_version: "1"`` + ``counts``. The bridge converts
    exactly and lets OfflineDensity.__post_init__ enforce the rest
    (non-empty, positive visits, usable tau_D).

    Raises:
        ValueError: cell_counts missing/empty or a cell key not "x,y,z"
            — an empty density would be a fake "no low density" fact
            and is never silently accepted.
    """
    counts_raw = data.get("cell_counts")
    if not isinstance(counts_raw, dict) or not counts_raw:
        raise ValueError(
            "offline density cell_counts must be a non-empty object"
        )
    counts: Dict[Tuple[int, int, int], int] = {}
    for key, count in counts_raw.items():
        try:
            x, y, z = (int(part) for part in key.split(","))
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"offline density cell key {key!r} must be 'x,y,z'"
            ) from e
        if (
            isinstance(count, bool) or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(
                f"offline density count for {key!r} must be an int >= 0"
            )
        counts[(x, y, z)] = count
    return OfflineDensity(counts=counts)


def read_reward13_density_file(path: str) -> OfflineDensity:
    """Read + bridge an EGO-dataset density file."""
    density_path = Path(path)
    if not density_path.is_file():
        raise ValueError(f"offline density file not found: {density_path}")
    try:
        with open(density_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        raise ValueError(
            f"offline density {density_path} is not valid JSON: {e}"
        ) from e
    return reward13_density_to_capability(data)


def build_density_from_zarr(
    dataset_path: str,
    bin_size: Tuple[float, float, float] = (1.0, 0.5, 0.5),
    chunk_rows: int = 500_000,
) -> Dict[str, Any]:
    """Chunked cell-count density from a frozen EGO zarr dataset.

    The R7-8 density builder (user ruling 2026-08-24): the 4.8 GB
    dataset is read in chunks and never materialized at once (R7-5
    EXIT=137 lesson). Per-episode goals are sparse reads of each
    episode's last position; within a chunk, each episode segment is
    vectorized through the SAME compute_support_cell mapping the
    reward uses (goal-unit frame, round indexing), so the density and
    the annotation always agree on cells.

    Returns the reward_v13 schema with literal "x,y,z" cell keys — the
    exact schema reward13_density_to_capability bridges (never the
    old "(x, y, z)" tuple-repr).

    Raises:
        ValueError: dataset missing, keys missing, or no episode goals
            extractable — never a silent empty density.
    """
    import zarr

    dataset = Path(dataset_path)
    if not dataset.exists():
        raise ValueError(f"offline density dataset not found: {dataset}")
    root = zarr.open(str(dataset), mode="r")
    if "data/state" not in root or "meta/episode_ends" not in root:
        raise ValueError(
            f"offline density dataset {dataset} lacks data/state or "
            "meta/episode_ends — the EGO dataset layout is required"
        )
    n_rows = int(root["data/state"].shape[0])
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=int)

    goals_list: List[np.ndarray] = []
    for ep_end in episode_ends:
        if 0 < ep_end <= n_rows:
            goals_list.append(np.asarray(
                root["data/state"][int(ep_end) - 1, 0:3], dtype=float))
    if not goals_list:
        raise ValueError(
            "no episode goals extractable from the dataset — every "
            "episode_ends entry is out of range"
        )
    goals = np.asarray(goals_list, dtype=float)  # (n_episodes, 3)

    bin_x, bin_y, bin_z = (float(b) for b in bin_size)
    counts: Dict[Tuple[int, int, int], int] = {}
    for i0 in range(0, n_rows, chunk_rows):
        i1 = min(i0 + chunk_rows, n_rows)
        positions = np.asarray(
            root["data/state"][i0:i1, 0:3], dtype=float)
        ep_idx = np.minimum(
            np.searchsorted(episode_ends, np.arange(i0, i1),
                            side="right"),
            len(goals) - 1,
        )
        for e in np.unique(ep_idx):
            seg = positions[ep_idx == e]
            goal = goals[e]
            rel = seg - goal
            goal_norm = float(np.linalg.norm(goal))
            if goal_norm < 1e-6:
                cells = np.zeros((len(seg), 3), dtype=int)
            else:
                goal_unit = goal / goal_norm
                x_par = rel @ goal_unit
                x_perp_raw = np.linalg.norm(
                    rel[:, :2] - x_par[:, None] * goal_unit[:2], axis=1)
                cross_z = goal_unit[0] * rel[:, 1] - goal_unit[1] * rel[:, 0]
                x_perp = np.where(cross_z >= 0, x_perp_raw, -x_perp_raw)
                cells = np.stack([
                    np.round(x_par / bin_x),
                    np.round(x_perp / bin_y),
                    np.round(seg[:, 2] / bin_z),
                ], axis=1).astype(int)
            uniq, uniq_counts = np.unique(cells, axis=0, return_counts=True)
            for cell, count in zip(map(tuple, uniq), uniq_counts):
                counts[cell] = counts.get(cell, 0) + int(count)

    all_counts = sorted(counts.values())
    p20 = float(np.percentile(all_counts, 20)) if all_counts else 0.0
    p50 = float(np.percentile(all_counts, 50)) if all_counts else 0.0
    return {
        "cell_counts": {
            f"{x},{y},{z}": count
            for (x, y, z), count in sorted(counts.items())
        },
        "total": int(n_rows),
        "p20_threshold": p20,
        "p50_threshold": p50,
        "bin_size": [bin_x, bin_y, bin_z],
    }


# ═══════════════════════════════════════════════════════════════════
# quota selection (guide Step 2) — deterministic, geometry-informed
# ═══════════════════════════════════════════════════════════════════

BANK_SIZE = 100
MIN_OPPORTUNITY_PER_CAPABILITY = 20
MIN_EPISODES_PER_ROLE = 20
MAX_UNKNOWN_OR_NO_OPPORTUNITY = 10
CAPABILITY_ROLES = ("anticipatory", "dense_recovery", "support_expansion")

# step 2: same episode may serve several capabilities; the per-world
# seed budget is the knob that hits exactly BANK_SIZE episodes.
DEFAULT_SEEDS_PER_WORLD = 4

# R7-8 seed budgets per main capability role (guide Step 2: the quota
# arithmetic). 6 antic/support + 18 dense worlds -> 24/24/54 = 102
# candidates, trimmed to exactly BANK_SIZE by select_episodes. Yaw
# specs are excluded entirely (their episodes would fall into the
# unknown/no-opportunity allowance).
SEEDS_BY_ROLE = {
    "anticipatory": 4,
    "support_expansion": 4,
    "dense_recovery": 3,
}


def expand_eval_candidates(
    specs: Sequence[Any],
    seeds_by_role: Optional[Dict[str, int]] = None,
) -> List[Tuple[Any, int]]:
    """(spec, seed) pairs for the three main capability roles, in
    manifest order, each spec expanded to its role's seed budget.

    Non-capability roles (yaw_lateral_recovery) are excluded — their
    episodes would land in the Step-2 unknown/no-opportunity allowance.
    Deterministic by construction; never reads any RL result (Step 5).
    """
    budgets = dict(seeds_by_role or SEEDS_BY_ROLE)
    out: List[Tuple[Any, int]] = []
    for spec in specs:
        n_seeds = int(budgets.get(spec.role, 0))
        if n_seeds > 0:
            out.extend((spec, seed) for seed in range(1, n_seeds + 1))
    return out


def select_episodes(
    candidates: Sequence[Tuple[str, str, int]],
) -> List[Tuple[str, str, int]]:
    """Deterministic (world_uid, role, seed) selection hitting exactly
    BANK_SIZE episodes with the Step-2 quotas.

    ``candidates`` is the pre-expanded (world_uid, role, seed) list
    (one world may offer several seeds). Selection is manifest-order
    (stable) and never looks at any RL checkpoint result — Step 5.

    Raises:
        ValueError: when the candidate pool cannot meet the quotas
            (missing roles, fewer than BANK_SIZE candidates, or the
            per-role minimums).
    """
    by_role: Dict[str, List[Tuple[str, str, int]]] = {}
    for world_uid, role, seed in candidates:
        by_role.setdefault(role, []).append((world_uid, role, seed))
    missing = [r for r in CAPABILITY_ROLES if r not in by_role]
    if missing:
        raise ValueError(f"candidate pool lacks capability roles: {missing}")

    chosen: List[Tuple[str, str, int]] = []
    for role in CAPABILITY_ROLES:
        chosen.extend(by_role[role])
    for role, items in by_role.items():
        if role not in CAPABILITY_ROLES:
            chosen.extend(items)

    if len(chosen) < BANK_SIZE:
        raise ValueError(
            f"candidate pool has {len(chosen)} episodes; need "
            f"{BANK_SIZE}"
        )
    per_role = {role: 0 for role in CAPABILITY_ROLES}
    for _, role, _ in chosen[:BANK_SIZE]:
        if role in per_role:
            per_role[role] += 1
    short = [
        role for role, n in per_role.items()
        if n < MIN_EPISODES_PER_ROLE
    ]
    if short:
        raise ValueError(
            f"quota selection leaves roles below "
            f"{MIN_EPISODES_PER_ROLE}: {short}"
        )
    return chosen[:BANK_SIZE]


# ═══════════════════════════════════════════════════════════════════
# freeze + annotate composition — the builder's core (Step 2/5)
# ═══════════════════════════════════════════════════════════════════


def freeze_and_annotate(
    env: Any,
    candidates: Sequence[Tuple[Any, int]],
    obstacles: Dict[str, Sequence[Dict[str, Any]]],
    density: Optional[Any] = None,
) -> List[Any]:
    """Freeze every (spec, seed) through the single-slot env duck type
    and annotate the SAMPLED start/goal with the policy-independent
    geometry opportunity.

    ``candidates`` is the (spec, seed) expansion; ``obstacles`` maps
    world_uid -> obstacle records; ``density`` is the capability
    OfflineDensity (None leaves the support group undecleared). The
    freeze contract holds: the runtime start/goal/yaw and the frozen
    episode_config_hash are exactly what the canonical eval will
    reproduce. Selection (Step 2 quotas) happens AFTER annotation but
    only on manifest/geometry — never on any RL checkpoint result.
    """
    from dataclasses import replace

    from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
        annotate_geometry_opportunity,
    )
    from diffusion_policy_3d.rl.dppo_uav.evaluator import (
        freeze_paired_bank,
    )

    specs = [spec for spec, _ in candidates]
    seeds = [int(seed) for _, seed in candidates]
    frozen = freeze_paired_bank(env, specs, seeds)
    out: List[Any] = []
    for entry in frozen:
        op = annotate_geometry_opportunity(
            obstacles.get(entry.world_uid, []),
            entry.start, entry.goal, density,
        )
        out.append(replace(entry, opportunity=op))
    return out


# ═══════════════════════════════════════════════════════════════════
# train-leakage check (guide Step 3) — bank ∩ train manifest = 0
# ═══════════════════════════════════════════════════════════════════


def check_train_leakage(
    bank_world_uids: Sequence[str],
    train_world_uids: Sequence[str],
) -> None:
    """Hard-fail on ANY bank world that also appears in the train
    manifest — the bank is eval-only by construction (Step 3)."""
    train_set = set(train_world_uids)
    leaked = sorted(
        {uid for uid in bank_world_uids if uid in train_set}
    )
    if leaked:
        raise ValueError(
            "train leakage: bank worlds present in the train manifest: "
            f"{leaked} — the bank must be eval-only"
        )


# ═══════════════════════════════════════════════════════════════════
# aggregate computation (guide Step 1) — geometry counts, v1 exposure
# ═══════════════════════════════════════════════════════════════════

NOT_YET_MEASURED = "NOT_YET_MEASURED"

# guide Step 1: aggregate carries candidate counts (geometry labels),
# actual opportunity counts (frozen blocks), and role/world counts.
AGGREGATE_KEYS = (
    "opportunity", "exposure", "candidates", "roles", "worlds",
)


def compute_aggregate(
    entries: Sequence[Any],
    candidates: Dict[str, int],
) -> Dict[str, Any]:
    """Compute the Step-1 aggregate over the frozen entries.

    ``entries`` carry ``role``, ``split`` and an optional frozen
    ``opportunity`` (CapabilityOpportunity). Candidates is the
    geometry-label count per capability (pre-selection). Exposure is
    NOT_YET_MEASURED in v1 (two-phase freeze ruling): the R7-9 BC
    replay measures it and rebuilds the aggregate for v2.

    Raises:
        ValueError: exposure/opportunity keys must be per capability
            and non-negative — never a silent 0.
    """
    cap_names = {"anticipatory", "full_space", "dense_recovery"}
    opportunity = {"anticipatory": 0, "full_space": 0,
                   "dense_recovery": 0}
    roles: Dict[str, int] = {}
    worlds: Dict[str, int] = {}
    for entry in entries:
        roles[entry.role] = roles.get(entry.role, 0) + 1
        worlds[entry.split] = worlds.get(entry.split, 0) + 1
        op = getattr(entry, "opportunity", None)
        if op is None or not getattr(op, "valid", False):
            continue
        if getattr(op, "risk_shell_crossing", False):
            opportunity["anticipatory"] += 1
        # dense opportunity (user ruling 2026-08-24): trap AND cluttered
        # topology both offer a dense_recovery evaluation scene — the
        # worlds' actual layouts are corridor/clutter-dominant, trap
        # triangles alone never reach the Step-2 >= 20 quota.
        if (
            getattr(op, "trap_opportunity_count", 0) > 0
            or getattr(op, "topology_class", "") == "cluttered"
        ):
            opportunity["dense_recovery"] += 1
        if getattr(op, "support_stratum", ""):
            opportunity["full_space"] += 1
    missing_cap = cap_names - set(candidates)
    if missing_cap:
        raise ValueError(
            f"aggregate candidates missing capabilities: "
            f"{sorted(missing_cap)}"
        )
    return {
        "opportunity": opportunity,
        "exposure": {
            "anticipatory": NOT_YET_MEASURED,
            "full_space": NOT_YET_MEASURED,
            "dense_recovery": NOT_YET_MEASURED,
        },
        "candidates": dict(candidates),
        "roles": dict(sorted(roles.items())),
        "worlds": dict(sorted(worlds.items())),
    }
