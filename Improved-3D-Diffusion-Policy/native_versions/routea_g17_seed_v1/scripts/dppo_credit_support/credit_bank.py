"""Frozen held-out-reset bank for paired DDPO credit diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path


SCHEMA = "dppo_credit_dev_bank_v2"
SAMPLING_CONTRACT = "stable_task_identity_training_seed_domain_heldout_reset_seed_v2"
MAX_TASKS = 1000
MAX_BANK_ENTRIES = 100000
MAX_EXCLUDED_TRAINING_SEGMENTS = 20000
MAX_BANK_BYTES = 64 * 1024 * 1024


def _field(item, name):
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)


def _task_identity_record(item):
    record = {
        "task_id": str(_field(item, "task_id")),
        "world_id": str(_field(item, "world_id")),
        "role": str(_field(item, "role")),
    }
    obstacle_hash = None
    if isinstance(item, Mapping):
        obstacle_hash = item.get("obstacle_sha256")
    if obstacle_hash is not None:
        record["obstacle_sha256"] = str(obstacle_hash)
    return record


def _task_seed(item) -> int:
    return int(
        _field(item, "task_seed") if isinstance(item, Mapping) else _field(item, "seed")
    )


def _seed_domain_hash(seed_domain) -> str:
    serialized = json.dumps(
        list(seed_domain), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _canonical_without_hash(payload) -> bytes:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _content_hash(payload) -> str:
    return hashlib.sha256(_canonical_without_hash(payload)).hexdigest()


def _derive_seed(namespace: str, purpose: str, index: int, task_id: str, nonce: int) -> int:
    material = "{}|{}|{}|{}|{}".format(
        namespace, purpose, index, task_id, nonce
    ).encode("utf-8")
    # uint32 works across NumPy/Gym versions and is accepted by torch.manual_seed.
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little")


def _validate_sha256(name: str, value: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return value


def make_dev_bank(
    tasks,
    *,
    bank_id: str,
    entry_count: int,
    source_manifest_sha256: str,
    exclude_training_segments: int,
):
    """Build a deterministic task-uniform bank with held-out reset seeds."""

    bank_id = str(bank_id)
    if not bank_id or len(bank_id) > 128:
        raise ValueError("bank_id must contain 1..128 characters")
    if not 0 < entry_count <= MAX_BANK_ENTRIES:
        raise ValueError(
            "entry_count must be within 1..{}".format(MAX_BANK_ENTRIES)
        )
    if not 0 <= exclude_training_segments <= MAX_EXCLUDED_TRAINING_SEGMENTS:
        raise ValueError(
            "exclude_training_segments must be within 0..{}".format(
                MAX_EXCLUDED_TRAINING_SEGMENTS
            )
        )
    manifest_hash = _validate_sha256(
        "source_manifest_sha256", source_manifest_sha256
    )
    task_items = tuple(tasks)
    ordered = sorted(
        (_task_identity_record(task) for task in task_items),
        key=lambda row: (row["role"], row["world_id"], row["task_id"]),
    )
    seed_domain = sorted(_task_seed(task) for task in task_items)
    if not ordered:
        raise ValueError("tasks must be non-empty")
    if len(ordered) > MAX_TASKS:
        raise ValueError("task count exceeds {}".format(MAX_TASKS))
    task_ids = [row["task_id"] for row in ordered]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_id values must be unique")
    if any(seed < 0 for seed in seed_domain):
        raise ValueError("task_seed values must be non-negative")
    if len(set(seed_domain)) != len(seed_domain):
        raise ValueError("task_seed values must be unique")

    excluded = {
        task_seed + segment * 1000003
        for task_seed in seed_domain
        for segment in range(exclude_training_segments + 1)
    }
    used_reset_seeds = set()
    used_policy_seeds = set()
    entries = []
    for index in range(entry_count):
        task = ordered[index % len(ordered)]
        nonce = 0
        while True:
            reset_seed = _derive_seed(bank_id, "reset", index, task["task_id"], nonce)
            if reset_seed not in excluded and reset_seed not in used_reset_seeds:
                break
            nonce += 1
        used_reset_seeds.add(reset_seed)
        nonce = 0
        while True:
            policy_seed = _derive_seed(
                bank_id, "stochastic_policy", index, task["task_id"], nonce
            )
            if policy_seed not in used_policy_seeds:
                break
            nonce += 1
        used_policy_seeds.add(policy_seed)
        entries.append(
            {
                "entry_index": index,
                "task_id": task["task_id"],
                "world_id": task["world_id"],
                "role": task["role"],
                "reset_seed": reset_seed,
                "policy_seed": policy_seed,
            }
        )

    role_counts = Counter(row["role"] for row in entries)
    payload = {
        "schema": SCHEMA,
        "bank_id": bank_id,
        "evidence_scope": "development_only_not_fixed32_not_formal_training",
        "sampling_contract": SAMPLING_CONTRACT,
        "source_manifest_sha256": manifest_hash,
        "source_task_count": len(ordered),
        "entry_count": len(entries),
        "exclude_training_segments_inclusive": exclude_training_segments,
        "training_reset_seed_overlap_count": 0,
        "training_task_seed_domain": seed_domain,
        "training_task_seed_domain_sha256": _seed_domain_hash(seed_domain),
        "source_tasks": ordered,
        "entry_role_counts": dict(sorted(role_counts.items())),
        "entries": entries,
    }
    payload["content_sha256"] = _content_hash(payload)
    return payload


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_nonfinite_json(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


def load_dev_bank(path: Path, *, expected_manifest_sha256: str, expected_tasks):
    """Load and bind a bank to the current manifest-derived task identities."""

    path = Path(path)
    if path.stat().st_size > MAX_BANK_BYTES:
        raise ValueError("dev bank exceeds the 64 MiB input limit")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("invalid dev bank JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unexpected dev bank schema")
    recorded_hash = str(payload.get("content_sha256", ""))
    if recorded_hash != _content_hash(payload):
        raise ValueError("dev bank content hash mismatch")
    manifest_hash = _validate_sha256(
        "expected_manifest_sha256", expected_manifest_sha256
    )
    if payload.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("dev bank manifest hash mismatch")
    if payload.get("sampling_contract") != SAMPLING_CONTRACT:
        raise ValueError("unexpected dev bank sampling contract")
    if payload.get("training_reset_seed_overlap_count") != 0:
        raise ValueError("dev bank overlaps an excluded training reset seed")

    expected_task_items = tuple(expected_tasks)
    expected = {
        row["task_id"]: row
        for row in (_task_identity_record(task) for task in expected_task_items)
    }
    expected_seed_domain = sorted(_task_seed(task) for task in expected_task_items)
    recorded_seed_domain = payload.get("training_task_seed_domain")
    if (
        not isinstance(recorded_seed_domain, list)
        or any(type(seed) is not int or seed < 0 for seed in recorded_seed_domain)
        or len(set(recorded_seed_domain)) != len(recorded_seed_domain)
    ):
        raise ValueError("dev bank training task seed domain is invalid")
    if recorded_seed_domain != expected_seed_domain:
        raise ValueError("dev bank training task seed domain mismatch")
    if payload.get("training_task_seed_domain_sha256") != _seed_domain_hash(
        expected_seed_domain
    ):
        raise ValueError("dev bank training task seed domain hash mismatch")
    recorded_tasks = payload.get("source_tasks")
    if not isinstance(recorded_tasks, list):
        raise ValueError("dev bank source_tasks must be a list")
    recorded = {row.get("task_id"): row for row in recorded_tasks if isinstance(row, dict)}
    if len(recorded) != len(recorded_tasks) or set(recorded) != set(expected):
        raise ValueError("dev bank task identity set mismatch")
    for task_id, expected_row in expected.items():
        recorded_row = recorded[task_id]
        for key in ("task_id", "world_id", "role"):
            if recorded_row.get(key) != expected_row.get(key):
                raise ValueError("dev bank task identity mismatch: {}".format(task_id))
        if "obstacle_sha256" in expected_row and recorded_row.get(
            "obstacle_sha256"
        ) != expected_row["obstacle_sha256"]:
            raise ValueError("dev bank obstacle hash mismatch: {}".format(task_id))

    entry_count = payload.get("entry_count")
    exclusion_bound = payload.get("exclude_training_segments_inclusive")
    if (
        not isinstance(entry_count, int)
        or isinstance(entry_count, bool)
        or not 0 < entry_count <= MAX_BANK_ENTRIES
        or not isinstance(exclusion_bound, int)
        or isinstance(exclusion_bound, bool)
        or not 0 <= exclusion_bound <= MAX_EXCLUDED_TRAINING_SEGMENTS
    ):
        raise ValueError("dev bank resource bounds are invalid")
    rebuilt = make_dev_bank(
        expected_task_items,
        bank_id=str(payload.get("bank_id", "")),
        entry_count=entry_count,
        source_manifest_sha256=manifest_hash,
        exclude_training_segments=exclusion_bound,
    )
    if payload != rebuilt:
        raise ValueError("dev bank does not match the deterministic builder contract")

    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != payload.get("entry_count"):
        raise ValueError("dev bank entry count mismatch")
    reset_seeds = set()
    policy_seeds = set()
    for index, row in enumerate(entries):
        if not isinstance(row, dict) or row.get("entry_index") != index:
            raise ValueError("dev bank entry order mismatch")
        task = expected.get(row.get("task_id"))
        if task is None or row.get("world_id") != task["world_id"] or row.get(
            "role"
        ) != task["role"]:
            raise ValueError("dev bank entry task mismatch")
        reset_seed = row.get("reset_seed")
        policy_seed = row.get("policy_seed")
        if (
            not isinstance(reset_seed, int)
            or isinstance(reset_seed, bool)
            or reset_seed < 0
            or not isinstance(policy_seed, int)
            or isinstance(policy_seed, bool)
            or policy_seed < 0
        ):
            raise ValueError("dev bank seeds must be non-negative integers")
        if reset_seed in reset_seeds or policy_seed in policy_seeds:
            raise ValueError("dev bank seeds must be unique")
        reset_seeds.add(reset_seed)
        policy_seeds.add(policy_seed)
    return payload
