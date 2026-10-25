import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_routea_credit_counterfactual import (
    D2CError,
    _validate_runtime_identity,
    main,
)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lane(
    root,
    name,
    *,
    bank_hash="bank",
    status="NO_MATERIAL_CREDIT_ROTATION",
    base_checkpoint_hash="base",
    u0_zero_effect=True,
):
    directory = root / name
    directory.mkdir(parents=True)
    summary = directory / "lane_summary.json"
    _write_json(
        summary,
        {
            "schema": "routea_d2c_lane_summary_v1",
            "lane": name,
            "bank_content_sha256": bank_hash,
            "transition_count": 1024,
            "pair_count": 10240,
            "status": status,
            "best_variant": "C2-finite-c10" if "ROTATION" in status else None,
            "variants": {},
            "tripwire_all_unchanged": True,
            "base_checkpoint_sha256": base_checkpoint_hash,
            "u0_zero_effect": u0_zero_effect if name.endswith("-u0") else None,
        },
    )
    (directory / "manifest_sha256.txt").write_text(
        "{}  lane_summary.json\n".format(_sha256(summary)), encoding="utf-8"
    )
    return directory


def test_self_check_preserves_frozen_training_source_hash(capsys):
    assert main(["--self-check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SELF_CHECK_PASS"
    assert payload["training_source_set_hash"] == (
        "5aed4fb675591750ac97ad4165ac11b8d3fedf5a15dfa2bf0bc84266c373f637"
    )
    assert payload["lanes"] == [
        "seed1-u0",
        "seed1-u32",
        "seed2-u0",
        "seed2-u32",
        "seed3-u0",
        "seed3-u32",
    ]


def test_finalize_requires_exact_six_lane_set_and_bank_matched_pairs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for lane in ("seed1-u0", "seed1-u32", "seed2-u0", "seed2-u32", "seed3-u0"):
        _lane(runs, lane, bank_hash=lane.split("-")[0])
    output = tmp_path / "final"
    assert main(["finalize-d2c", "--runs-root", str(runs), "--output-dir", str(output)]) == 10
    assert not output.exists()

    _lane(runs, "seed3-u32", bank_hash="different")
    assert main(["finalize-d2c", "--runs-root", str(runs), "--output-dir", str(output)]) == 10
    assert not output.exists()


def test_finalize_reports_cross_seed_scope_without_authorizing_training(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for seed in ("seed1", "seed2", "seed3"):
        _lane(runs, seed + "-u0", bank_hash=seed)
        _lane(
            runs,
            seed + "-u32",
            bank_hash=seed,
            status="LOCAL_ACTOR_CREDIT_ROTATION_CREDIT",
        )
    output = tmp_path / "final"
    assert main(["finalize-d2c", "--runs-root", str(runs), "--output-dir", str(output)]) == 0
    verdict = json.loads((output / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "CROSS_SEED_LOCAL_ACTOR_CREDIT_ROTATION_CREDIT"
    assert verdict["training_authorized"] is False
    assert verdict["causal_scope"] == "frozen_critic_fixed_trajectory_local_actor_credit"
    assert (output / "manifest_sha256.txt").is_file()


def test_capture_cli_rejects_checkpoint_for_u0(tmp_path):
    with pytest.raises(SystemExit) as captured:
        main(
            [
                "capture-d2c",
                "--lane",
                "seed1-u0",
                "--config",
                str(tmp_path / "config.yaml"),
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--dev-bank",
                str(tmp_path / "bank.json"),
                "--output-dir",
                str(tmp_path / "out"),
                "--expected-source-set-hash",
                "0" * 64,
            ]
        )
    assert captured.value.code == 2


def test_finalize_rejects_nonshared_bc_or_nonzero_effect_u0(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for seed in ("seed1", "seed2", "seed3"):
        _lane(runs, seed + "-u0", bank_hash=seed)
        _lane(runs, seed + "-u32", bank_hash=seed)
    _write_json(
        runs / "seed2-u0" / "lane_summary.json",
        {
            **json.loads((runs / "seed2-u0" / "lane_summary.json").read_text(encoding="utf-8")),
            "u0_zero_effect": False,
        },
    )
    summary = runs / "seed2-u0" / "lane_summary.json"
    (runs / "seed2-u0" / "manifest_sha256.txt").write_text(
        "{}  lane_summary.json\n".format(_sha256(summary)), encoding="utf-8"
    )
    output = tmp_path / "final"
    assert main(["finalize-d2c", "--runs-root", str(runs), "--output-dir", str(output)]) == 11
    assert not output.exists()


def test_finalize_rejects_stale_extra_lane_directory(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for seed in ("seed1", "seed2", "seed3"):
        _lane(runs, seed + "-u0", bank_hash=seed)
        _lane(runs, seed + "-u32", bank_hash=seed)
    (runs / "stale-old-run").mkdir()
    output = tmp_path / "final"
    assert main(["finalize-d2c", "--runs-root", str(runs), "--output-dir", str(output)]) == 10
    assert not output.exists()


def test_run_identity_rejects_checkpoint_changed_after_capture():
    captured = {
        "base_checkpoint_sha256": "base-a",
        "checkpoint_sha256": "rl-a",
    }
    with pytest.raises(D2CError, match="base BC checkpoint changed"):
        _validate_runtime_identity(
            captured,
            lane="seed1-u32",
            base_checkpoint_sha256="base-b",
            checkpoint_sha256="rl-a",
        )
    with pytest.raises(D2CError, match="RL checkpoint changed"):
        _validate_runtime_identity(
            captured,
            lane="seed1-u32",
            base_checkpoint_sha256="base-a",
            checkpoint_sha256="rl-b",
        )


def test_run_cli_rejects_microbatch_that_cannot_match_official_reference(tmp_path):
    with pytest.raises(SystemExit) as captured:
        main(
            [
                "run-d2c",
                "--lane",
                "seed1-u0",
                "--config",
                str(tmp_path / "config.yaml"),
                "--capture-root",
                str(tmp_path / "captures"),
                "--output-dir",
                str(tmp_path / "out"),
                "--expected-source-set-hash",
                "0" * 64,
                "--microbatch-pairs",
                "64",
            ]
        )
    assert captured.value.code == 2
