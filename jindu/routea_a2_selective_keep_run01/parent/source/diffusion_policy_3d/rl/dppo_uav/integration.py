"""Real-env integration entry.

This module is the single construction point the gates (G5/G6) and
formal training use:

- build_world_pool: the frozen scan_world_pool semantics (env_*.obstacles.json
  glob, name-sorted; kinematic_uav_env_v2.py:56);
- build_routea_train_pool: the PRODUCTION pool semantics (frozen E040
  build_envs, train_ddpm_ppo_meanlogp_3_e040.py L690-734): admission
  manifest rl_train buckets -> worlds/train/<uid>.world.obstacles.json
  (routeA_rl_v1 naming). The frozen env_* glob does NOT match the
  production pool — the manifest-driven builder is what G3/G5/G6 use;
- entry_metadata_hash: deterministic MANIFEST SOURCE hash
  (manifest_entry_hash) over the entry's fixed recorded config — the
  runtime SAMPLED start/goal/yaw are covered by episode_config_hash at
  reset instead);
- build_adapter: UAVVectorEnvAdapter over that pool, carrying composite-key
  manifest records (world_uid, role, admission_id) into the AdmissionChecker
  (rule 6);
- build_task_specs: TaskSpecRegistry -> (train_specs, eval_specs).

AdmissionChecker + A* planner live inside the adapter; train_loop
consumes the duck-typed env contract unchanged.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from diffusion_policy_3d.env.kinematic_uav_env_v2 import (
    TRAIN_WORLDS_DIR,
    scan_world_pool,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSpec,
    TaskSpecRegistry,
    UAVVectorEnvAdapter,
    canonical_sha256,
)
from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
    TRAINABLE_ROLES,
    build_role_buckets,
    load_routea_rl_admission_manifest,
)

_ROLE_ORDER = (
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
)


def build_world_pool(worlds_dir: str) -> List[Tuple[str, str]]:
    """[(world_uid, obstacles_json), ...] — frozen scan_world_pool glob."""
    return scan_world_pool(worlds_dir)


def _routea_pool(
    manifest: Any,
    worlds_roots: Dict[str, str],
    splits: Sequence[str],
) -> List[Tuple[str, str]]:
    """[(world_uid, obstacles_json), ...] over the given manifest splits.

    Mirrors the frozen E040 build_envs semantics
    (train_ddpm_ppo_meanlogp_3_e040.py L690-734): admission manifest
    buckets, per role, file-existence checked, de-duplicated. Entries
    with a missing obstacles file are skipped (E040: ``if
    os.path.exists(json_path) and name not in seen``).

    R7-8 (user ruling 2026-08-24): each split's world files are looked
    up in ITS OWN directory (rl_train -> worlds/train, rl_val ->
    worlds/val, rl_test -> worlds/test) — "rl_eval" is the normative
    name for rl_val ∪ rl_test and the literal splits resolve
    independently. A split without a mapped root contributes nothing
    (two-arg callers keep the old train-only behavior).
    """
    pool: List[Tuple[str, str]] = []
    seen = set()
    for split in splits:
        root = worlds_roots.get(split)
        if root is None:
            continue
        buckets = build_role_buckets(manifest, split)
        for role in _ROLE_ORDER:
            for entry in buckets.get(role, []):
                uid = entry.world_uid
                if uid in seen:
                    continue
                json_path = os.path.join(
                    root, f"{uid}.world.obstacles.json"
                )
                if os.path.isfile(json_path):
                    pool.append((uid, json_path))
                    seen.add(uid)
    return pool


def build_routea_train_pool(
    manifest_path: str,
    train_worlds_dir: str,
) -> List[Tuple[str, str]]:
    """[(world_uid, obstacles_json), ...] for rl_train admitted worlds.

    The TRAINING pool — rl_train buckets only (eval-only worlds must
    never enter training; the sampler boundary stays closed, P1-1).
    """
    manifest = load_routea_rl_admission_manifest(manifest_path)
    return _routea_pool(
        manifest, {"rl_train": train_worlds_dir}, ("rl_train",)
    )


def build_routea_eval_pool(
    manifest_path: str,
    train_worlds_dir: str,
    val_worlds_dir: Optional[str] = None,
    test_worlds_dir: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """[(world_uid, obstacles_json), ...] for admitted rl_train ∪
    rl_val ∪ rl_test worlds.

    The canonical EVAL pool (P1-1, design v1.3 '只评估不训练'): eval-only
    worlds (w4/w8/w11/w12/w18 style) must be reachable by the canonical
    eval — the old rl_train-only pool excluded them structurally.
    R7-8: "rl_eval" has no literal manifest split; the eval pool is the
    three literal splits resolved per-split (worlds/{train,val,test}).
    val/test dirs are optional — callers without them keep the old
    train-only behavior (val/test entries are skipped, never crash).
    """
    manifest = load_routea_rl_admission_manifest(manifest_path)
    roots = {"rl_train": train_worlds_dir}
    if val_worlds_dir is not None:
        roots["rl_val"] = val_worlds_dir
    if test_worlds_dir is not None:
        roots["rl_test"] = test_worlds_dir
    return _routea_pool(
        manifest, roots, ("rl_train", "rl_val", "rl_test")
    )


def _flatten_entries(
    manifest: Any, splits: Sequence[str],
) -> List[Dict[str, Any]]:
    """Admitted manifest entries with a resolved ``role`` field.

    build_role_buckets filters ``role_active or role_validated`` against
    TRAINABLE_ROLES and ``admission_status.startswith("admitted")``, so
    the audited worlds are exactly the production pools. Single flatten
    shared by the train/eval callers.
    """
    entries: List[Dict[str, Any]] = []
    for split in splits:
        buckets = build_role_buckets(manifest, split)
        for role in TRAINABLE_ROLES:
            items = buckets[role]
            for item in items:
                d: Dict[str, Any] = dict(vars(item))
                d.pop("raw", None)
                d.update(item.raw)  # raw manifest fields win (start/goal, ...)
                d["role"] = role
                entries.append(d)
    return entries


def manifest_train_entries(manifest_path: str) -> List[Dict[str, Any]]:
    """Admitted rl_train manifest entries with a resolved ``role`` field.

    The train pool (build_routea_train_pool), manifest records, and task
    specs all derive from the SAME buckets — the audited worlds are
    exactly the production training pool. Used by G3/G5/G6 callers.
    """
    manifest = load_routea_rl_admission_manifest(manifest_path)
    return _flatten_entries(manifest, ("rl_train",))


def manifest_eval_entries(manifest_path: str) -> List[Dict[str, Any]]:
    """Admitted rl_train ∪ rl_val ∪ rl_test manifest entries with a
    resolved ``role`` field.

    The canonical eval pool and its task specs derive from the SAME
    buckets (P1-1): eval-only '只评估不训练' worlds are included — the
    eval side evaluates what the training side refuses to train on.
    R7-8: "rl_eval" is the normative name for rl_val ∪ rl_test (the
    manifest carries the three literal splits).
    """
    manifest = load_routea_rl_admission_manifest(manifest_path)
    return _flatten_entries(manifest, ("rl_train", "rl_val", "rl_test"))


def entry_metadata_hash(entry: Dict[str, Any]) -> str:
    """Deterministic sha256 over the entry's fixed recorded config.

    This is the MANIFEST SOURCE hash (manifest_entry_hash) over the entry's
    recorded fields — it does NOT cover the runtime SAMPLED start/goal/yaw
    (episode_config_hash, computed at reset, covers those). No recorded hash
    exists in the manifest or
    worldgen sidecars, so the hash is derived from the fixed fields the
    entry actually records. Canonical serialization: a non-JSON field
    (e.g. numpy array) hard-fails instead of default=str swallowing it.

    The canonical source entry is the FULL raw manifest entry — every fixed
    field, with NO whitelist (a whitelist would cover only a handful of the
    ~25 production fields, so a change to any unlisted column —
    admission_status, bc_fm_*, ego_*, n_episodes_*, pass2/stop/calibration
    fields, ... — would be invisible to rule-6 and to the G3 hash gates). A
    NEW manifest column binds immediately. Added on top: the RESOLVED
    fields spec construction actually consumes — role (the field
    manifest_train_entries injects, role_active fallback), admission_id /
    generator_version (TaskSpecRegistry defaults), and the ACTUAL obstacle
    path the environment loads (<uid>.world.obstacles.json — the frozen
    E040 build_envs path; a plain f"{uid}.json" default would record a file
    that is never loaded, so the hash would bind a phantom input). The only
    excluded key is the hash itself (hashing it would be a cycle).
    """
    payload = {
        k: v for k, v in entry.items() if k != "manifest_entry_hash"
    }
    uid = entry.get("world_uid", "")
    role = entry.get("role") or entry.get("role_active") or ""
    payload["resolved_role"] = role
    payload["resolved_admission_id"] = (
        entry.get("admission_id")
        or f"{uid}__{role}__{entry.get('split', '')}"
    )
    payload["resolved_generator_version"] = (
        entry.get("generator_version") or "routeA_rl_v1.3"
    )
    payload["resolved_obstacle_path"] = (
        entry.get("obstacle_json") or f"{uid}.world.obstacles.json"
    )
    return canonical_sha256(payload)


def build_adapter(
    contract: RLSystemContract,
    *,
    worlds_dir: str = str(TRAIN_WORLDS_DIR),
    manifest_path: Optional[str] = None,
    manifest_records: Optional[Dict[str, str]] = None,
    device: str = "cpu",
    pool: Optional[List[Tuple[str, str]]] = None,
    **kwargs,
) -> UAVVectorEnvAdapter:
    """Real training env over the world pool.

    ``manifest_path`` selects the PRODUCTION pool semantics — admission
    manifest rl_train buckets -> worlds/train/<uid>.world.obstacles.json
    (frozen E040 build_envs; G3/G5/G6 use this). Without it, the pool is
    the frozen v2 scan_world_pool glob (env_*). ``pool`` (P1-1): an
    explicit eval pool wins over both — the canonical eval hands its own
    rl_train ∪ rl_eval pool here. ``manifest_records`` feeds
    the AdmissionChecker rule 6, keyed by manifest_record_key(world_uid,
    role, admission_id) (composite key — two roles sharing a world_uid
    never overwrite each other) — a spec with a stale source hash exhausts
    instead of admitting. Remaining kwargs pass through to
    UAVVectorEnvAdapter (max_episode_steps, curriculum_stage,
    disturb_enable, ...).
    """
    if pool is None:
        if manifest_path is not None:
            pool = build_routea_train_pool(manifest_path, worlds_dir)
        else:
            pool = build_world_pool(worlds_dir)
    return UAVVectorEnvAdapter(
        contract,
        world_pool=pool,
        device=device,
        manifest_records=manifest_records,
        **kwargs,
    )


def build_task_specs(
    entries: List[Dict[str, str]],
) -> Tuple[List[EpisodeTaskSpec], List[EpisodeTaskSpec]]:
    """(train_specs, eval_specs) from manifest entries (TaskSpecRegistry)."""
    registry = TaskSpecRegistry(entries)
    return registry.train_specs(), registry.eval_specs()
