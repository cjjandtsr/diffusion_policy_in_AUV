#!/usr/bin/env python3
"""aggregator.py — P0 capacity budget audit aggregator.

Reads seven sub-report JSONs from the output directory, compiles gate results,
and produces the total-gate decision dict.

Sub-report contract (REQUIRED fields in each sub-report JSON)
==============================================================
    schema_version (str):  Version string for the sub-report schema.
    run_id         (str):  Run identifier matching the parent audit run.
    gate_pass      (bool): Whether this individual gate passed.

Sub-register that do not conform (missing required fields, unreadable,
or absent) produce a gate = False.

Total-gate schema (produced by run_capacity_budget_p0_audit)
=============================================================
    schema_version     (str):  Hard-coded schema version string.
    run_id             (str):  From config, or derived from output_dir basename.
    gates              (dict): Map of seven gate_key -> bool.
    overall_pass       (bool): All seven gates true and no guard refusal.
    training_authorized(bool): Strictly equal to overall_pass.
    artifact_hashes    (dict): Map of sub-report filename -> sha256 hex digest.
    refused_pass_reason(str, optional): Set when overall_pass is forcibly false.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────

SUB_REPORT_MAP: Dict[str, str] = {
    "artifact_identity": "artifact_hashes.json",
    "lora_scaling": "lora_scaling_audit.json",
    "resume_state": "resume_state_audit.json",
    "eval_bank": "eval_bank_audit.json",
    "reward_wiring": "reward_wiring_audit.json",
    "role_distribution": "role_distribution_audit.json",
    "multicheckpoint_schema": "multicheckpoint_schema_audit.json",
}

SCHEMA_VERSION = "routea_capacity_budget_p0_v1"

GATE_PASS_KEY = "gate_pass"
"""Canonical key name for the pass/fail field in sub-report JSONs.

Downstream Tasks MUST use this constant rather than hardcoding the string."""

SUB_REPORT_CONTRACT = {
    "schema_version": "str — Version string for the sub-report schema.",
    "run_id": "str — Run identifier matching the parent audit run.",
    GATE_PASS_KEY: "bool — Whether this individual gate passed.",
}
"""Sub-report contract: every sub-report JSON MUST contain these three fields."""


# ── Internal helpers ────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    """Return hex sha256 digest of *path* (read in 64 KiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sub_report(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Read a sub-report JSON and return ``(gate_pass, error_label_or_None)``.

    Enforces the FULL sub-report contract: the JSON must be a dict containing
    ``schema_version`` (any value is accepted — presence only),
    ``run_id`` (presence only), and ``GATE_PASS_KEY`` which must be a
    Python ``bool`` (not a truthy string or integer).

    ``error_label`` is one of:
        * ``"file_missing"`` — file does not exist or is not a regular file.
        * ``"unreadable"`` — JSON decode error or OS I/O error.
        * ``"not_a_dict"`` — JSON parsed but is not a dict (e.g. a list).
        * ``"missing_schema_version"`` — JSON lacks the ``schema_version`` key.
        * ``"missing_run_id"`` — JSON lacks the ``run_id`` key.
        * ``"missing_gate_pass"`` — JSON lacks the ``gate_pass`` key.
        * ``"gate_pass_not_bool"`` — ``gate_pass`` present but not a Python bool.
    """
    if not filepath.is_file():
        return False, "file_missing"
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, "unreadable"
    if not isinstance(data, dict):
        return False, "not_a_dict"
    if "schema_version" not in data:
        return False, "missing_schema_version"
    if "run_id" not in data:
        return False, "missing_run_id"
    if GATE_PASS_KEY not in data:
        return False, "missing_gate_pass"
    if not isinstance(data[GATE_PASS_KEY], bool):
        return False, "gate_pass_not_bool"
    return data[GATE_PASS_KEY], None


# ── Public entry point ──────────────────────────────────────────────────────


def run_capacity_budget_p0_audit(config: dict) -> dict:
    """Return a JSON-serializable dict containing seven sub-gates and overall_pass.

    Parameters
    ----------
    config : dict
        Must contain at least:
            ``output_dir`` (str): Path to directory containing sub-report JSONs.
            ``run_id`` (str, optional):  Run identifier.  Derived from
                *output_dir* basename if absent.

    Returns
    -------
    dict
        Total-gate decision dict (see module docstring for schema).
        Also written to ``<output_dir>/p0_gate_decision.json`` if *output_dir*
        is configured.
    """
    output_dir_str: Optional[str] = config.get("output_dir")
    run_id: Optional[str] = config.get("run_id")

    refused_pass_reason: Optional[str] = None

    # ── Derive run_id ──────────────────────────────────────────────────────
    if not run_id:
        if output_dir_str:
            run_id = Path(output_dir_str).name
        else:
            refused_pass_reason = (
                "Cannot produce run_id: no 'run_id' in config and no "
                "'output_dir' from which to derive a basename."
            )

    output_dir: Optional[Path] = Path(output_dir_str) if output_dir_str else None

    # ── Read each sub-report ───────────────────────────────────────────────
    gates: Dict[str, bool] = {}
    artifact_hashes: Dict[str, str] = {}

    for gate_key, filename in SUB_REPORT_MAP.items():
        if output_dir is None:
            gates[gate_key] = False
            continue

        filepath = output_dir / filename
        gate_pass, error = _read_sub_report(filepath)
        gates[gate_key] = gate_pass

        if filepath.is_file():
            try:
                artifact_hashes[filename] = _sha256_file(filepath)
            except OSError:
                pass  # non-critical; omit hash for this file

    # ── Compute pass ───────────────────────────────────────────────────────
    overall_pass: bool = all(gates.values()) and refused_pass_reason is None

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or "unknown",
        "gates": gates,
        "overall_pass": overall_pass,
        "training_authorized": overall_pass,
        "artifact_hashes": artifact_hashes,
    }

    if refused_pass_reason is not None:
        result["overall_pass"] = False
        result["training_authorized"] = False
        result["refused_pass_reason"] = refused_pass_reason

    # Invariant: training_authorized MUST strictly equal overall_pass.
    assert result["training_authorized"] == result["overall_pass"], (
        f"Invariant violation: training_authorized ({result['training_authorized']}) "
        f"!= overall_pass ({result['overall_pass']})"
    )

    # ── Persist ────────────────────────────────────────────────────────────
    if output_dir is not None:
        out_path = output_dir / "p0_gate_decision.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result
