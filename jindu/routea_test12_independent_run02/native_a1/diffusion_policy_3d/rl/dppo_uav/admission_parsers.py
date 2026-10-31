"""R7-7 strict JSON parsers for every admission input.

Each parser:
- rejects NaN / Infinity at the JSON layer (json.loads is lax by
  default; parse_constant raises instead);
- rejects unknown schema_version and empty tables;
- checks the REQUIRED raw numeric/schema fields and raises
  ``ValidationError`` carrying the offending field name — the
  aggregator routes the message into the precise decision list
  (failed_requirement_ids / stale_artifacts / missing_artifacts /
  review_failures), so the anti-cheat tests can assert the exact
  fragment.

These schemas are the AUTHORITY the R7-9 gate producers must emit
(R7-7 KDD-5): ``status == \"PASS\"`` is never trusted here — only raw
numeric/schema fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .admission_rules import (
    SHA256_HEX_LEN,
    GIT_COMMIT_HEX_LEN,
    skip_reason_allowlisted,
)

HEX_RE = re.compile(r"^[0-9a-f]+$")

# Who may have produced an r7-test-result-v1.  Anything else (hand-written
# summaries, another harness) is rejected: the raw log is the audit
# ground-truth, and only pytest runs the suite.
TEST_RESULT_PRODUCERS = ("pytest",)

G3_HARD_GATES = (
    "invalid_window",
    "no_ray",
    "runtime_hash_mismatch",
    "source_hash_missing",
    "source_hash_mismatch",
    "eval_leakage",
    "unknown_role",
    "unvalidated_fallback",
    "admission_blocked",
    "manifest_hash_mismatch",
)
G3_N_GATES = 15
BANK_CAPABILITIES = ("anticipatory", "full_space", "dense_recovery")
BANK_MIN_OPPORTUNITY = 20
BANK_MIN_EXPOSURE = 20
G7_N_SEEDS = 2
G7_UPDATES_PER_SEED = 30


class ValidationError(ValueError):
    """One precise admission-input violation (message names the field)."""


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise ValidationError(f"{label} file missing: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(
                f,
                parse_constant=lambda tok: (_ for _ in ()).throw(
                    ValidationError(
                        f"{label}: JSON constant {tok!r} (NaN/Infinity) "
                        "is not allowed in admission inputs"
                    )
                ),
            )
    except ValidationError:
        raise
    except (OSError, ValueError) as e:
        raise ValidationError(f"{label}: unreadable/invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError(f"{label}: root must be a JSON object")
    return data


def _require(data: dict, key: str, label: str) -> None:
    if key not in data:
        raise ValidationError(
            f"{label}: missing required schema field {key!r}"
        )


def _check_hex(value, name: str, length: int) -> None:
    if not isinstance(value, str) or len(value) != length or not HEX_RE.match(
        value
    ):
        raise ValidationError(
            f"{name} must be {length}-char lowercase hex, got {value!r}"
        )


def _check_int(value, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got {value!r}")


def _check_bool(value, name: str) -> None:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a boolean, got {value!r}")


def _check_number(value, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a number, got {value!r}")


def _check_string(value, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} must be a non-empty string")


def verify_log_binding(
    result_dir: Path, log_path: str, log_sha256: str, label: str,
    *, required_texts: tuple[str, ...] | None = None,
) -> None:
    """A result is only credible when the raw log it names really exists
    and hashes to the declared sha256 (Zread P0-2).  Relative log_path
    is resolved against the result JSON's directory.  When
    ``required_texts`` is given, every text must appear in the log
    content (e.g. the GPU nodeids that must actually have run)."""
    p = Path(log_path)
    if not p.is_absolute():
        p = result_dir / p
    if not p.exists():
        raise ValidationError(f"{label}: log_path does not exist: {p}")
    raw = p.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != log_sha256:
        raise ValidationError(
            f"{label}: log sha256 mismatch for {p} (declared "
            f"{log_sha256[:12]}…, actual {actual[:12]}…)"
        )
    if required_texts:
        text = raw.decode("utf-8", errors="replace")
        missing = [t for t in required_texts if t not in text]
        if missing:
            raise ValidationError(
                f"{label}: log {p} does not contain required nodeid(s): "
                f"{missing}"
            )


def parse_test_result(path: Path) -> dict:
    """CPU/GPU full-suite result (schema r7-test-result-v1)."""
    label = f"test-result {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-test-result-v1":
        raise ValidationError(
            f"{label}: unknown schema_version "
            f"{data.get('schema_version')!r} (expected r7-test-result-v1)"
        )
    for key in (
        "command", "started_at", "finished_at", "exit_code", "collected",
        "passed", "failed", "error", "skipped", "xfail", "xpass",
        "deselected", "commit", "tree", "validated_source_set_hash",
        "producer", "log_path", "log_sha256", "skips",
    ):
        _require(data, key, label)
    _check_int(data["exit_code"], f"{label} exit_code")
    _check_int(data["collected"], f"{label} collected")
    for key in ("passed", "failed", "error", "skipped", "xfail", "xpass",
                "deselected"):
        _check_int(data[key], f"{label} {key}")
    if data["exit_code"] != 0:
        raise ValidationError(f"{label} exit_code must be 0")
    if data["failed"] != 0:
        raise ValidationError(
            f"{label} failed must be 0 (got {data['failed']})"
        )
    if data["error"] != 0:
        raise ValidationError(f"{label} error must be 0 (got {data['error']})")
    if data["collected"] < 1:
        raise ValidationError(f"{label} collected must be > 0")
    if data["collected"] != (
        data["passed"] + data["failed"] + data["error"]
        + data["skipped"] + data["xfail"] + data["xpass"]
    ):
        raise ValidationError(
            f"{label} collected does not equal passed+failed+error+skipped"
            "+xfail+xpass"
        )
    _check_hex(data["commit"], f"{label} commit", GIT_COMMIT_HEX_LEN)
    _check_hex(data["tree"], f"{label} tree", GIT_COMMIT_HEX_LEN)
    _check_hex(
        data["validated_source_set_hash"],
        f"{label} validated_source_set_hash", SHA256_HEX_LEN,
    )
    _check_string(data["producer"], f"{label} producer")
    if data["producer"] not in TEST_RESULT_PRODUCERS:
        raise ValidationError(
            f"{label}: producer {data['producer']!r} not on the frozen "
            f"allowlist {TEST_RESULT_PRODUCERS}"
        )
    _check_string(data["log_path"], f"{label} log_path")
    _check_hex(data["log_sha256"], f"{label} log_sha256", SHA256_HEX_LEN)
    verify_log_binding(
        path.parent, data["log_path"], data["log_sha256"], label)
    if not isinstance(data["skips"], list):
        raise ValidationError(f"{label} skips must be a list")
    for skip in data["skips"]:
        if not isinstance(skip, dict):
            raise ValidationError(f"{label}: skip entries must be objects")
        for key in ("nodeid", "reason", "owner", "unrelated_proof"):
            if key not in skip or not isinstance(skip[key], str) or not skip[
                key
            ]:
                raise ValidationError(
                    f"{label}: skip entry missing non-empty {key!r}"
                )
        if skip_reason_allowlisted(skip["reason"]) is None:
            raise ValidationError(
                f"{label}: skip not on the frozen allowlist: "
                f"{skip['reason']!r} ({skip['nodeid']})"
            )
    return data


def parse_g3_result(path: Path) -> dict:
    """10k-reset gate audit (schema r7-g3-result-v1)."""
    label = f"g3 {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-g3-result-v1":
        raise ValidationError(
            f"{label}: unknown schema_version {data.get('schema_version')!r}"
        )
    for key in ("n_resets", "role_proportion_error", "reachability_pass_ratio",
                "gates", "commit", "tree", "validated_source_set_hash"):
        _require(data, key, label)
    _check_int(data["n_resets"], f"{label} n_resets")
    if data["n_resets"] < 10000:
        raise ValidationError(
            f"{label} n_resets must be >= 10000 (got {data['n_resets']})"
        )
    for gate in G3_HARD_GATES:
        if gate not in data:
            raise ValidationError(f"{label}: missing hard gate {gate!r}")
        _check_int(data[gate], f"{label} {gate}")
        if data[gate] != 0:
            raise ValidationError(
                f"{label} {gate} must be 0 (got {data[gate]})"
            )
    _check_number(
        data["role_proportion_error"], f"{label} role_proportion_error"
    )
    if data["role_proportion_error"] > 2.0:
        raise ValidationError(
            f"{label} role_proportion_error must be <= 2.0 "
            f"(got {data['role_proportion_error']})"
        )
    _check_number(
        data["reachability_pass_ratio"], f"{label} reachability_pass_ratio"
    )
    if data["reachability_pass_ratio"] < 1.0:
        raise ValidationError(
            f"{label} reachability_pass_ratio must be >= 1.0 "
            f"(got {data['reachability_pass_ratio']})"
        )
    if not isinstance(data["gates"], dict) or len(data["gates"]) != G3_N_GATES:
        raise ValidationError(
            f"{label} gates must map exactly {G3_N_GATES} entries"
        )
    for name, status in data["gates"].items():
        if status != "PASS":
            raise ValidationError(
                f"{label} gate {name!r} must be PASS (got {status!r})"
            )
    _check_hex(data["commit"], f"{label} commit", GIT_COMMIT_HEX_LEN)
    _check_hex(data["tree"], f"{label} tree", GIT_COMMIT_HEX_LEN)
    _check_hex(
        data["validated_source_set_hash"],
        f"{label} validated_source_set_hash", SHA256_HEX_LEN,
    )
    return data


def parse_g5_result(path: Path) -> dict:
    """Effective-PPO identity / LoRA before-after (schema r7-g5-result-v1)."""
    label = f"g5 {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-g5-result-v1":
        raise ValidationError(
            f"{label}: unknown schema_version {data.get('schema_version')!r}"
        )
    for key in ("n_episodes", "b1_identity", "b128_identity",
                "lora_base_before_after", "numerical_mode", "commit", "tree",
                "validated_source_set_hash"):
        _require(data, key, label)
    _check_int(data["n_episodes"], f"{label} n_episodes")
    if data["n_episodes"] != 32:
        raise ValidationError(
            f"{label} n_episodes must be 32 (got {data['n_episodes']})"
        )
    b1 = data["b1_identity"]
    if not isinstance(b1, dict) or "max_abs_diff" not in b1:
        raise ValidationError(f"{label}: b1_identity must carry max_abs_diff")
    _check_number(b1["max_abs_diff"], f"{label} b1_identity.max_abs_diff")
    if b1["max_abs_diff"] > 1e-5:
        raise ValidationError(
            f"{label} b1_identity.max_abs_diff must be <= 1e-5 "
            f"(got {b1['max_abs_diff']})"
        )
    bb = data["lora_base_before_after"]
    if not isinstance(bb, dict):
        raise ValidationError(f"{label}: lora_base_before_after must be object")
    for key in ("base_unchanged", "lora_updated"):
        if key not in bb:
            raise ValidationError(
                f"{label}: lora_base_before_after missing {key!r}"
            )
        _check_bool(bb[key], f"{label} lora_base_before_after.{key}")
    if not bb["base_unchanged"]:
        raise ValidationError(
            f"{label} base must be unchanged (base_unchanged false)"
        )
    if not bb["lora_updated"]:
        raise ValidationError(f"{label} lora must be updated (lora_updated false)")
    _check_hex(data["commit"], f"{label} commit", GIT_COMMIT_HEX_LEN)
    _check_hex(data["tree"], f"{label} tree", GIT_COMMIT_HEX_LEN)
    _check_hex(
        data["validated_source_set_hash"],
        f"{label} validated_source_set_hash", SHA256_HEX_LEN,
    )
    return data


def parse_g6_result(path: Path) -> dict:
    """One-update GPU gradient/optimizer audit (schema r7-g6-result-v1)."""
    label = f"g6 {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-g6-result-v1":
        raise ValidationError(
            f"{label}: unknown schema_version {data.get('schema_version')!r}"
        )
    for key in ("actor_grad", "critic_grad", "lora_base_update",
                "unexecuted_grad", "ratio", "optimizer_step", "resume",
                "commit", "tree", "validated_source_set_hash"):
        _require(data, key, label)
    actor = data["actor_grad"]
    if not isinstance(actor, dict) or not isinstance(actor.get("nonzero"), bool):
        raise ValidationError(f"{label}: actor_grad.nonzero must be a boolean")
    if not actor["nonzero"]:
        raise ValidationError(f"{label} actor grad must be nonzero")
    critic = data["critic_grad"]
    if not isinstance(critic, dict) or not isinstance(
        critic.get("nonzero"), bool
    ):
        raise ValidationError(f"{label}: critic_grad.nonzero must be a boolean")
    if not critic["nonzero"]:
        raise ValidationError(f"{label} critic grad must be nonzero")
    lb = data["lora_base_update"]
    if not isinstance(lb, dict) or not isinstance(lb.get("base_unchanged"), bool):
        raise ValidationError(
            f"{label}: lora_base_update.base_unchanged must be a boolean"
        )
    if not lb["base_unchanged"]:
        raise ValidationError(f"{label} base must stay unchanged in g6")
    ueg = data["unexecuted_grad"]
    if not isinstance(ueg, dict) or "count" not in ueg:
        raise ValidationError(f"{label}: unexecuted_grad.count required")
    _check_int(ueg["count"], f"{label} unexecuted_grad.count")
    if ueg["count"] != 0:
        raise ValidationError(
            f"{label} unexecuted_grad.count must be 0 (got {ueg['count']})"
        )
    ratio = data["ratio"]
    if not isinstance(ratio, dict) or not isinstance(ratio.get("finite"), bool):
        raise ValidationError(f"{label}: ratio.finite must be a boolean")
    if not ratio["finite"]:
        raise ValidationError(f"{label} ratio must be finite")
    opt = data["optimizer_step"]
    if not isinstance(opt, dict) or not isinstance(opt.get("occurred"), bool):
        raise ValidationError(f"{label}: optimizer_step.occurred must be boolean")
    if not opt["occurred"]:
        raise ValidationError(f"{label} optimizer_step must have occurred")
    rsm = data["resume"]
    if not isinstance(rsm, dict) or not isinstance(rsm.get("worked"), bool):
        raise ValidationError(f"{label}: resume.worked must be a boolean")
    if not rsm["worked"]:
        raise ValidationError(f"{label} resume must have worked")
    _check_hex(data["commit"], f"{label} commit", GIT_COMMIT_HEX_LEN)
    _check_hex(data["tree"], f"{label} tree", GIT_COMMIT_HEX_LEN)
    _check_hex(
        data["validated_source_set_hash"],
        f"{label} validated_source_set_hash", SHA256_HEX_LEN,
    )
    return data


def parse_resume_result(path: Path) -> dict:
    """Same-seed resume comparator (schema r7-compare-v1, R7-6).

    F3 (Zread P1): must be bound to the REAL raw GPU pytest log — an
    allowlisted ``producer``, a log whose file exists with a matching
    sha256, and the GPU nodeid(s) that actually ran (each must appear in
    the log text, so a CPU-mock or GPU-skipped run cannot impersonate).
    """
    label = f"resume {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-compare-v1":
        raise ValidationError(
            f"{label}: unknown schema_version {data.get('schema_version')!r}"
        )
    for key in ("pass", "metric_diffs", "ckpt_diffs", "first_diff",
                "producer", "gpu_test_nodeids", "log_path", "log_sha256"):
        _require(data, key, label)
    _check_bool(data["pass"], f"{label} pass")
    _check_int(data["metric_diffs"], f"{label} metric_diffs")
    _check_int(data["ckpt_diffs"], f"{label} ckpt_diffs")
    if not data["pass"]:
        raise ValidationError(f"{label} pass must be true")
    if data["metric_diffs"] != 0:
        raise ValidationError(
            f"{label} metric_diffs must be 0 (got {data['metric_diffs']})"
        )
    if data["ckpt_diffs"] != 0:
        raise ValidationError(
            f"{label} ckpt_diffs must be 0 (got {data['ckpt_diffs']})"
        )
    _check_string(data["producer"], f"{label} producer")
    if data["producer"] not in TEST_RESULT_PRODUCERS:
        raise ValidationError(
            f"{label}: producer {data['producer']!r} not on the frozen "
            f"allowlist {TEST_RESULT_PRODUCERS}"
        )
    nodeids = data["gpu_test_nodeids"]
    if not isinstance(nodeids, list) or not nodeids:
        raise ValidationError(
            f"{label} gpu_test_nodeids must be a non-empty list"
        )
    for nid in nodeids:
        if not isinstance(nid, str) or not nid:
            raise ValidationError(
                f"{label}: gpu_test_nodeids entries must be non-empty "
                "strings"
            )
    _check_string(data["log_path"], f"{label} log_path")
    _check_hex(data["log_sha256"], f"{label} log_sha256", SHA256_HEX_LEN)
    verify_log_binding(
        path.parent, data["log_path"], data["log_sha256"], label,
        required_texts=tuple(nodeids),
    )
    return data


def parse_g7_result(path: Path) -> dict:
    """Two-seed exact-budget G7 run (schema r7-g7-result-v1)."""
    label = f"g7 {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-g7-result-v1":
        raise ValidationError(
            f"{label}: unknown schema_version {data.get('schema_version')!r}"
        )
    for key in ("seeds", "transitions_per_seed", "updates_per_seed",
                "hard_faults", "per_seed", "commit", "tree",
                "validated_source_set_hash"):
        _require(data, key, label)
    seeds = data["seeds"]
    if not isinstance(seeds, list) or len(seeds) != G7_N_SEEDS or len(
        set(seeds)
    ) != G7_N_SEEDS:
        raise ValidationError(
            f"{label} seeds must be exactly {G7_N_SEEDS} distinct seeds "
            f"(got {seeds!r})"
        )
    _check_int(data["updates_per_seed"], f"{label} updates_per_seed")
    if data["updates_per_seed"] != G7_UPDATES_PER_SEED:
        raise ValidationError(
            f"{label} updates_per_seed must be {G7_UPDATES_PER_SEED} "
            f"(got {data['updates_per_seed']})"
        )
    _check_int(data["hard_faults"], f"{label} hard_faults")
    if data["hard_faults"] != 0:
        raise ValidationError(
            f"{label} hard_faults must be 0 (got {data['hard_faults']})"
        )
    per_seed = data["per_seed"]
    if not isinstance(per_seed, dict) or set(per_seed) != set(
        str(s) for s in seeds
    ):
        raise ValidationError(
            f"{label} per_seed keys must equal the seeds exactly "
            f"(got {sorted(per_seed)})"
        )
    transitions = []
    for sid, rec in per_seed.items():
        if not isinstance(rec, dict):
            raise ValidationError(f"{label}: per_seed.{sid} must be an object")
        for key in ("transitions", "updates", "releases"):
            if key not in rec:
                raise ValidationError(
                    f"{label}: per_seed.{sid} missing {key!r}"
                )
            _check_int(rec[key], f"{label} per_seed.{sid}.{key}")
        transitions.append(rec["transitions"])
    if len(set(transitions)) != 1:
        raise ValidationError(
            f"{label} transitions must be EQUAL across the two seeds "
            f"(got {transitions})"
        )
    _check_hex(data["commit"], f"{label} commit", GIT_COMMIT_HEX_LEN)
    _check_hex(data["tree"], f"{label} tree", GIT_COMMIT_HEX_LEN)
    _check_hex(
        data["validated_source_set_hash"],
        f"{label} validated_source_set_hash", SHA256_HEX_LEN,
    )
    return data


def parse_bank_result(path: Path) -> dict:
    """Frozen paired bank aggregate (schema r7-paired-bank-v1)."""
    label = f"bank {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-paired-bank-v1":
        raise ValidationError(
            f"{label}: unknown schema_version {data.get('schema_version')!r}"
        )
    for key in ("bank_size", "membership_hash", "aggregate"):
        _require(data, key, label)
    _check_int(data["bank_size"], f"{label} bank_size")
    if data["bank_size"] < 100:
        raise ValidationError(
            f"{label} bank_size must be >= 100 (got {data['bank_size']})"
        )
    _check_hex(data["membership_hash"], f"{label} membership_hash",
               SHA256_HEX_LEN)
    agg = data["aggregate"]
    if not isinstance(agg, dict):
        raise ValidationError(f"{label}: aggregate must be an object")
    for axis in ("opportunity", "exposure"):
        if axis not in agg or not isinstance(agg[axis], dict):
            raise ValidationError(f"{label}: aggregate.{axis} required")
        for cap in BANK_CAPABILITIES:
            if cap not in agg[axis]:
                raise ValidationError(
                    f"{label}: aggregate.{axis} missing capability {cap!r}"
                )
            _check_int(agg[axis][cap], f"{label} aggregate.{axis}.{cap}")
            minimum = (
                BANK_MIN_OPPORTUNITY if axis == "opportunity"
                else BANK_MIN_EXPOSURE
            )
            if agg[axis][cap] < minimum:
                raise ValidationError(
                    f"{label} aggregate.{axis}.{cap} must be >= {minimum} "
                    f"(got {agg[axis][cap]})"
                )
    return data


def parse_review(path: Path) -> dict:
    """One review record (schema r7-review-v1)."""
    label = f"review {path}"
    data = _load_json(path, label)
    if data.get("schema_version") != "r7-review-v1":
        raise ValidationError(
            f"{label}: unknown schema_version {data.get('schema_version')!r}"
        )
    for key in ("kind", "status", "reviewer", "bound_commit", "bound_tree",
                "bound_source_set_hash", "artifact_hashes", "reviewed_at"):
        _require(data, key, label)
    _check_string(data["kind"], f"{label} kind")
    _check_string(data["status"], f"{label} status")
    _check_string(data["reviewer"], f"{label} reviewer")
    _check_hex(data["bound_commit"], f"{label} bound_commit",
               GIT_COMMIT_HEX_LEN)
    _check_hex(data["bound_tree"], f"{label} bound_tree", GIT_COMMIT_HEX_LEN)
    _check_hex(
        data["bound_source_set_hash"], f"{label} bound_source_set_hash",
        SHA256_HEX_LEN,
    )
    if not isinstance(data["artifact_hashes"], list):
        raise ValidationError(f"{label}: artifact_hashes must be a list")
    return data
