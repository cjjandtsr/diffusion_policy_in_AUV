#!/usr/bin/env python3
"""Fail-closed structural gate for paired BC/RL DDPO credit-probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_FILE_BYTES = 64 * 1024 * 1024


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_nonfinite_json(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


def _read_json(path: Path):
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("missing or oversized JSON: {}".format(path))
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON: {}".format(path)) from exc


def _read_jsonl(path: Path, expected_schema: str):
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("missing or oversized JSONL: {}".format(path))
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank JSONL row: {}:{}".format(path, line_number))
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_nonfinite_json,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSONL row: {}:{}".format(path, line_number)
                ) from exc
            if not isinstance(row, dict) or row.get("schema") != expected_schema:
                raise ValueError(
                    "unexpected JSONL schema: {}:{}".format(path, line_number)
                )
            rows.append(row)
    return rows


def _validate_one(
    root: Path,
    label: str,
    expected_transitions: int,
    expected_gradients: int,
    expected_source_set_hash: str,
):
    summary = _read_json(root / "summary.json")
    provenance = _read_json(root / "provenance.json")
    transitions = _read_jsonl(
        root / "transitions.jsonl", "dppo_credit_transition_v2"
    )
    gradients = _read_jsonl(
        root / "gradient_minibatches.jsonl", "dppo_credit_gradient_v1"
    )
    if summary.get("schema") != "dppo_credit_probe_summary_v2":
        raise ValueError("{} summary schema mismatch".format(label))
    if provenance.get("schema") != "dppo_credit_probe_provenance_v2":
        raise ValueError("{} provenance schema mismatch".format(label))
    if summary.get("status") != "PROBE_COMPLETE_NO_PARAMETER_UPDATE":
        raise ValueError("{} did not complete read-only probe".format(label))
    if summary.get("policy_label") != label or provenance.get("policy_label") != label:
        raise ValueError("{} policy label mismatch".format(label))
    if len(transitions) != expected_transitions or summary.get(
        "transitions"
    ) != expected_transitions:
        raise ValueError("{} transition count mismatch".format(label))
    if len(gradients) != expected_gradients or summary.get(
        "gradient_minibatches"
    ) != expected_gradients:
        raise ValueError("{} gradient minibatch count mismatch".format(label))
    parameter_delta = summary.get("parameter_delta", {})
    if parameter_delta.get("max_abs") != 0.0 or parameter_delta.get(
        "nonzero_elements"
    ) != 0:
        raise ValueError("{} actor parameter delta is nonzero".format(label))
    if not summary.get("training_counters_unchanged") or not summary.get(
        "parameter_gradients_unchanged"
    ):
        raise ValueError("{} mutated counters or gradients".format(label))
    if summary.get("training_call_tripwire_triggered") is not False:
        raise ValueError("{} training tripwire triggered".format(label))
    if provenance.get("training_enabled") is not False or provenance.get(
        "optimizer_step_called"
    ) is not False:
        raise ValueError("{} provenance permits training".format(label))
    if provenance.get("training_call_tripwire_armed") is not True:
        raise ValueError("{} training tripwire was not armed".format(label))
    for hash_field in (
        "dev_bank_content_sha256",
        "config_sha256",
        "base_bc_checkpoint_sha256",
        "script_sha256",
        "credit_module_sha256",
        "credit_bank_module_sha256",
        "expected_source_set_hash",
        "actual_source_set_hash",
    ):
        value = provenance.get(hash_field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("{} {} is not a SHA-256".format(label, hash_field))
    if provenance.get("expected_source_set_hash") != expected_source_set_hash:
        raise ValueError("{} expected source-set binding mismatch".format(label))
    if provenance.get("actual_source_set_hash") != expected_source_set_hash:
        raise ValueError("{} executable source-set mismatch".format(label))
    bank_hash = summary.get("dev_bank_content_sha256")
    if bank_hash != provenance.get("dev_bank_content_sha256"):
        raise ValueError("{} dev bank binding mismatch".format(label))

    keys = set()
    for row in transitions:
        if row.get("policy_label") != label:
            raise ValueError("{} transition policy label mismatch".format(label))
        entry = row.get("bank_entry_index")
        decision = row.get("decision_index")
        if (
            not isinstance(entry, int)
            or isinstance(entry, bool)
            or entry < 0
            or not isinstance(decision, int)
            or isinstance(decision, bool)
            or decision < 0
        ):
            raise ValueError("{} transition pairing key is invalid".format(label))
        key = (entry, decision)
        if key in keys:
            raise ValueError("{} transition pairing key is duplicated".format(label))
        keys.add(key)
    return summary, provenance, keys


def validate_pair(args):
    bc_summary, bc_provenance, bc_keys = _validate_one(
        args.bc_dir.resolve(),
        "bc",
        args.expected_transitions,
        args.expected_gradient_minibatches,
        args.expected_source_set_hash,
    )
    rl_summary, rl_provenance, rl_keys = _validate_one(
        args.rl_dir.resolve(),
        "rl",
        args.expected_transitions,
        args.expected_gradient_minibatches,
        args.expected_source_set_hash,
    )
    for field, description in (
        ("dev_bank_content_sha256", "dev bank"),
        ("config_sha256", "config"),
        ("base_bc_checkpoint_sha256", "base BC checkpoint"),
        ("script_sha256", "probe script"),
        ("credit_module_sha256", "credit module"),
        ("credit_bank_module_sha256", "credit bank module"),
        ("expected_source_set_hash", "expected source set"),
        ("actual_source_set_hash", "executable source set"),
    ):
        if bc_provenance.get(field) != rl_provenance.get(field):
            raise ValueError("paired {} identity mismatch".format(description))
    if bc_summary.get("dev_bank_content_sha256") != rl_summary.get(
        "dev_bank_content_sha256"
    ):
        raise ValueError("paired dev bank summary mismatch")
    if bc_provenance.get("checkpoint_update") != 0:
        raise ValueError("BC probe is not the fresh update-0 state")
    if rl_provenance.get("checkpoint_update") != args.expected_rl_update:
        raise ValueError("RL checkpoint update mismatch")
    common = bc_keys.intersection(rl_keys)
    if not common:
        raise ValueError("paired probes have no common bank-entry decisions")
    result = {
        "schema": "dppo_credit_pair_gate_v1",
        "status": "PAIR_GATE_PASS",
        "dev_bank_content_sha256": bc_provenance["dev_bank_content_sha256"],
        "config_sha256": bc_provenance["config_sha256"],
        "base_bc_checkpoint_sha256": bc_provenance[
            "base_bc_checkpoint_sha256"
        ],
        "bc_transitions": len(bc_keys),
        "rl_transitions": len(rl_keys),
        "common_bank_entry_decisions": len(common),
        "expected_rl_update": args.expected_rl_update,
        "source_set_hash": args.expected_source_set_hash,
        "training_or_parameter_update_observed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-dir", required=True, type=Path)
    parser.add_argument("--rl-dir", required=True, type=Path)
    parser.add_argument("--expected-transitions", required=True, type=int)
    parser.add_argument(
        "--expected-gradient-minibatches", required=True, type=int
    )
    parser.add_argument("--expected-rl-update", type=int, default=256)
    parser.add_argument("--expected-source-set-hash", required=True)
    args = parser.parse_args(argv)
    if args.expected_transitions <= 0:
        parser.error("--expected-transitions must be positive")
    if args.expected_gradient_minibatches < 0:
        parser.error("--expected-gradient-minibatches must be non-negative")
    if args.expected_rl_update <= 0:
        parser.error("--expected-rl-update must be positive")
    try:
        if len(args.expected_source_set_hash) != 64:
            raise ValueError
        int(args.expected_source_set_hash, 16)
    except ValueError:
        parser.error("--expected-source-set-hash must be 64 lowercase hex characters")
    if args.expected_source_set_hash.lower() != args.expected_source_set_hash:
        parser.error("--expected-source-set-hash must be 64 lowercase hex characters")
    return validate_pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
