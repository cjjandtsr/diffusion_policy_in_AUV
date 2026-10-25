"""R3.3 (LEGACY-FORMAL-GUARD + canonical entry preflight).

Legacy side: every `scripts/train_ddpm_ppo*.py` entry point must reject
`--mode formal` with the HISTORICAL rejection in a subprocess — formal
training runs only through the A-Official-Core path
(`scripts/train_dppo_uav_official.py`), never through a legacy E-series
script (RED history: the frozen E040 file carried no guard and exited 0).

Canonical side: `scripts/train_dppo_uav_official.py` and
`scripts/eval_dppo_uav_official.py` must hard-fail their preflight
(git clean/commit, vendor hashes, BC/manifest hashes, projection=False,
train/eval split, numerical mode) BEFORE any env/optimizer construction.
Exit-code contract: 1 = preflight/identity failure, 2 = CLI-level
failure (argparse, bank contract). Positive preflight paths require a
clean git tree, so they are only verifiable post-commit (the entries
refuse to run in a dirty tree by design) — the two clean-tree tests
skip on a dirty tree and go live after the commit. The eval bank checks
(pool reachability, manifest hash truth) run before preflight
(CLI-level, exit 2).
"""

import atexit
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    EMPTY_LORA_CHECKSUM,
    SCHEMA_VERSION,
    CheckpointBuilder,
    CheckpointData,
    config_hash_of,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    TRACE_STEP_REQUIRED_KEYS,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    canonical_sha256,
)
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
    EVAL_ONLY_WORLDS,
    HISTORICAL_INVALID_FOR_FORMAL_TRAINING,
    assert_no_eval_only_worlds,
    official_source_file_hashes,
    verify_vendor_hashes,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_routea_train_pool,
    entry_metadata_hash,
    manifest_train_entries,
)

LEGACY_TRAIN_SCRIPTS = sorted(glob.glob("scripts/train_ddpm_ppo*.py"))
TRAIN_OFFICIAL = "scripts/train_dppo_uav_official.py"
EVAL_OFFICIAL = "scripts/eval_dppo_uav_official.py"

# The production assets the canonical entries bind to (their CLI defaults).
REAL_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
REAL_WORLDS_TRAIN = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
REAL_BC_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    "/checkpoints/latest.ckpt"
)

needs_real_assets = pytest.mark.skipif(
    not (
        os.path.isfile(REAL_MANIFEST)
        and os.path.isdir(REAL_WORLDS_TRAIN)
        and os.path.isfile(REAL_BC_CKPT)
    ),
    reason="production manifest/worlds/BC assets not present",
)


@pytest.fixture(scope="module")
def _utf8_env() -> dict:
    """Force UTF-8 stdio in the subprocess (the guard message carries an
    em-dash; LANG=C would decode the captured output as ascii)."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _preflight_clean() -> bool:
    """git status --porcelain shows only temp/-prefixed untracked files."""
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if out.returncode != 0:
        return False
    return all(
        ln.startswith("?? temp/") for ln in out.stdout.splitlines() if ln
    )


def _write_dirty_marker() -> str:
    """Untracked file OUTSIDE temp/ — the exact preflight-forbidden state.

    Unique per process (a leftover marker from a crashed run never
    collides) and atexit-removed (a hard kill cannot leave the tree
    permanently dirty — a leftover marker would block the canonical
    entries until manually removed).
    """
    marker = f".__r3_preflight_dirty_marker_{os.getpid()}"
    atexit.register(lambda: os.path.exists(marker) and os.remove(marker))
    with open(marker, "w") as f:
        f.write("dirty-tree marker for the R3.3 preflight test\n")
    return marker


def _real_bank_payload() -> dict:
    """One manifest-admitted, pool-reachable entry as a valid FROZEN bank
    payload (bank v3: the bank_id binds the full content byte-for-byte —
    the CLI refuses a v2 bank without one)."""
    pool = build_routea_train_pool(REAL_MANIFEST, REAL_WORLDS_TRAIN)
    pool_uids = {uid for uid, _ in pool}
    entry = next(
        e for e in manifest_train_entries(REAL_MANIFEST)
        if e["world_uid"] in pool_uids
    )
    bank = {
        "bank_version": "3",
        "source_hash": "0" * 64,
        "creation_commit": "0" * 40,
        "entries": [{
            "world_uid": entry["world_uid"],
            "role": entry["role"],
            "split": entry["split"],
            # Resolved composite key — the same fallback the spec
            # registry applies when the manifest entry lacks the raw
            # admission_id field.
            "admission_id": entry.get("admission_id")
            or f"{entry['world_uid']}__{entry['role']}__{entry['split']}",
            "manifest_entry_hash": entry_metadata_hash(entry),
            "obstacle_json": f"{entry['world_uid']}.world.obstacles.json",
            "generator_version": "routeA_rl_v1.3",
            "seed": 1001,
            "start": [1.0, 1.0, 1.0],
            "goal": [5.0, 5.0, 2.0],
            "start_yaw": 0.0,
            # Structurally valid only — these canonical tests never reach
            # the eval-time freeze check (they exit at CLI level).
            "episode_config_hash": "0" * 64,
            "disturbance": {"events": []},
            "admission": {
                "start_clearance": 1.0, "goal_clearance": 1.0,
                "reachability_pass": True, "path_length": 0.0,
                "path_stretch": 0.0,
            },
            # Guide R7-8 Step 1 (required_trace_fields schema, 2026-08-24):
            # every frozen record declares the trace contract; the strict
            # bank loader refuses a record without it.
            "required_trace_fields": list(TRACE_STEP_REQUIRED_KEYS),
        }],
    }
    # Mirror paired_bank_to_dict: bank_id = canonical sha256 over the
    # content EXCLUDING the injected bank_id (bank_version is IN the
    # content — only the self-referencing bank_id is excluded),
    # injected top-level AND per-entry.
    bank["bank_id"] = canonical_sha256(
        {k: v for k, v in bank.items() if k != "bank_id"}
    )
    for entry_dict in bank["entries"]:
        entry_dict["bank_id"] = bank["bank_id"]
        entry_dict["bank_version"] = "3"
    return bank


def _real_density_payload() -> dict:
    """A structurally valid offline density field (schema_version 1).

    The canonical tests exit at CLI level — only the schema matters, the
    counts never reach a capability computation.
    """
    return {
        "schema_version": "1",
        "bin_size": [1.0, 0.5, 0.5],
        "counts": {"0,0,2": 5, "1,0,2": 5},
    }


def _fake_checkpoint(path: str) -> None:
    """A structurally valid schema-v3 checkpoint with a WRONG identity.

    Rework #2 (P1-3): v3 requires lora_identity + environment_rng_state
    (30 fields total) — the old v2 fixture crashed CheckpointData
    construction (TypeError) before the subprocess could ever reach the
    identity-rejection path, so the test it fed never actually ran. The
    config_layer is self-consistent (config_hash recomputes to it) so the
    divergence surfaces as a runtime identity mismatch on the real asset
    hashes, exactly the path the test intends to exercise.
    """
    layer = {
        "device": "cpu",
        "reward_version": "v1.3",
        "loop_mode": "one_update",
        "n_episodes": 2,
        "max_ep_len": 12,
        "eval_n_episodes": 0,
        "gate_phase": "rl",
        "min_sampling_denoising_std": 0.1,
        "min_logprob_denoising_std": 0.1,
        "numerical_mode": {"tf32": False, "deterministic": True},
        "policy_contract": {"policy_cfg_lambda": 1.5},
        "trainer": {"update_epochs": 1, "batch_size": 4},
        "training_identity": {"base_seed": 0, "n_envs": 1},
    }
    data = CheckpointData(
        schema_version=SCHEMA_VERSION,
        official_dppo_commit="0" * 40,
        project_git_commit="0" * 40,
        project_tree_hash="0" * 40,
        source_file_hashes={},
        contract_serialized=b"{}",
        contract_hash="0" * 64,
        manifest_hash="0" * 64,
        bc_checkpoint_hash="0" * 64,
        config_hash=config_hash_of(layer),
        policy_state={},
        value_state={},
        lora_state={},
        lora_identity={
            "lora_rank": 0, "lora_alpha": 0.0, "n_adaptors": 0,
            "n_lora_param_tensors": 0,
            "lora_checksum": EMPTY_LORA_CHECKSUM,
        },
        actor_optimizer_state={},
        value_optimizer_state={},
        lr_scheduler_state={},
        sampling_std_state={},
        reward_norm_state={},
        rng_states={},
        numerical_mode_state={},
        environment_rng_state={
            "seed_source": "sampler_state",
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
            "restorable": True,
        },
        sampler_state={},
        outer_iteration=0,
        ppo_update_count=0,
        optimizer_step_count=0,
        environment_transition_count=1,
        completed_episode_count=0,
        wall_clock_timestamp="2026-08-10T00:00:00",
        exact_resolved_config={"config_layer": layer},
    )
    CheckpointBuilder.save(data, Path(path))


@pytest.mark.parametrize("script", LEGACY_TRAIN_SCRIPTS)
def test_legacy_script_rejects_formal_mode_subprocess(script, _utf8_env):
    """Every legacy E-series script must exit non-zero on --mode formal
    with the HISTORICAL rejection — a pass-through is a bypass of the
    unique-entry rule (E040 was the RED gap: it exited 0)."""
    r = subprocess.run(
        [sys.executable, script, "--dry_run", "--mode", "formal"],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=180,
    )
    assert r.returncode != 0, (
        f"{script} accepted --mode formal (exit 0) — formal training "
        "must be rejected at a legacy entry point"
    )
    assert HISTORICAL_INVALID_FOR_FORMAL_TRAINING in (r.stdout + r.stderr), (
        f"{script} failed for the wrong reason — expected the HISTORICAL "
        "rejection message"
    )


def test_legacy_scripts_enumerated():
    """The parametrized coverage must span every legacy train script."""
    assert len(LEGACY_TRAIN_SCRIPTS) >= 7, LEGACY_TRAIN_SCRIPTS
    assert "scripts/train_ddpm_ppo_meanlogp_3_e040.py" in LEGACY_TRAIN_SCRIPTS


# ═══════════════════════════════════════════════════════════════════
# canonical entries — failure paths + pure validation functions
# (positive preflight paths need a clean tree: post-commit only)
# ═══════════════════════════════════════════════════════════════════

def test_train_official_rejects_unknown_stage(_utf8_env):
    """argparse rejects an unknown stage (exit 2) before any preflight."""
    r = subprocess.run(
        [sys.executable, TRAIN_OFFICIAL, "--stage", "bogus",
         "--output_dir", "temp/x"],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=120,
    )
    assert r.returncode == 2, f"argparse exit 2 expected, got {r.returncode}"


def test_train_official_preflight_fails_on_dirty_tree(_utf8_env):
    """A non-clean tree hard-fails the canonical train entry (exit 1).

    The marker is an untracked file OUTSIDE temp/ — the exact condition
    the preflight forbids. It is created and removed within the test
    (unique per process, atexit-guarded against a killed run).
    """
    marker = _write_dirty_marker()
    try:
        r = subprocess.run(
            [sys.executable, TRAIN_OFFICIAL, "--stage", "no_update",
             "--output_dir", "temp/x"],
            capture_output=True, text=True, encoding="utf-8",
            env=_utf8_env, timeout=120,
        )
    finally:
        if os.path.exists(marker):
            os.remove(marker)
    assert r.returncode == 1, (
        f"canonical train must exit 1 on a dirty tree, got {r.returncode}"
    )
    assert "preflight FAIL" in (r.stdout + r.stderr), (
        "expected the preflight rejection, got: "
        + (r.stdout + r.stderr)[-500:]
    )


def test_eval_official_requires_checkpoint_and_bank(_utf8_env):
    """--checkpoint/--paired_bank are required (argparse exit 2)."""
    r = subprocess.run(
        [sys.executable, EVAL_OFFICIAL, "--output_dir", "temp/x"],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=120,
    )
    assert r.returncode == 2, f"argparse exit 2 expected, got {r.returncode}"


def test_eval_official_missing_bank_fails(_utf8_env, tmp_path):
    """A missing paired bank is a CLI-level hard failure (exit 2, before
    preflight)."""
    r = subprocess.run(
        [sys.executable, EVAL_OFFICIAL,
         "--checkpoint", str(tmp_path / "none.pt"),
         "--paired_bank", str(tmp_path / "no_bank.json"),
         "--offline_density", str(tmp_path / "no_density.json"),
         "--output_dir", str(tmp_path / "out")],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=120,
    )
    assert r.returncode == 2, (
        f"missing paired bank must exit 2, got {r.returncode}"
    )
    assert "paired bank" in (r.stdout + r.stderr), (
        "expected the bank error, got: " + (r.stdout + r.stderr)[-500:]
    )


def test_eval_official_missing_density_fails(_utf8_env, tmp_path):
    """A missing offline density is a CLI-level hard failure (exit 2,
    before preflight) — capability metrics cannot degrade to zero."""
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(_real_bank_payload()))
    r = subprocess.run(
        [sys.executable, EVAL_OFFICIAL,
         "--checkpoint", str(tmp_path / "none.pt"),
         "--paired_bank", str(bank),
         "--offline_density", str(tmp_path / "no_density.json"),
         "--output_dir", str(tmp_path / "out")],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=120,
    )
    assert r.returncode == 2, (
        f"missing offline density must exit 2, got {r.returncode}"
    )
    assert "offline density" in (r.stdout + r.stderr), (
        "expected the density error, got: " + (r.stdout + r.stderr)[-500:]
    )


@pytest.mark.parametrize("payload,note", [
    ("not json at all", "unparseable JSON"),
    ('{"bank_version": "1"}', "bank_version schema mismatch"),
    ('{"bank_version": "2"}', "missing source_hash"),
    ('{"bank_version": "2", "source_hash": "s", "creation_commit": "c",'
     ' "entries": []}', "empty entries"),
    ('{"bank_version": "2", "source_hash": "s", "creation_commit": "c",'
     ' "entries": [{"world_uid": "w1"}]}',
     "entry missing required fields"),
])
def test_eval_official_bad_bank_fails(payload, note, _utf8_env, tmp_path):
    """Structurally invalid banks are rejected at CLI level (exit 2)."""
    bank = tmp_path / "bad_bank.json"
    bank.write_text(payload)
    r = subprocess.run(
        [sys.executable, EVAL_OFFICIAL,
         "--checkpoint", str(tmp_path / "none.pt"),
         "--paired_bank", str(bank),
         "--offline_density", str(tmp_path / "no_density.json"),
         "--output_dir", str(tmp_path / "out")],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=120,
    )
    assert r.returncode == 2, (
        f"bad bank ({note}) must exit 2, got {r.returncode}"
    )
    assert "bank" in (r.stdout + r.stderr), (
        f"bad bank ({note}) failed for the wrong reason: "
        + (r.stdout + r.stderr)[-500:]
    )


@needs_real_assets
def test_eval_official_preflight_fails_on_dirty_tree(_utf8_env, tmp_path):
    """Eval preflight is the same hard gate (exit 1) — the bank must first
    pass the CLI checks (valid bank), then the dirty tree fails preflight."""
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(_real_bank_payload()))
    density = tmp_path / "density.json"
    density.write_text(json.dumps(_real_density_payload()))
    marker = _write_dirty_marker()
    try:
        r = subprocess.run(
            [sys.executable, EVAL_OFFICIAL,
             "--checkpoint", str(tmp_path / "none.pt"),
             "--paired_bank", str(bank),
             "--offline_density", str(density),
             "--output_dir", str(tmp_path / "out")],
            capture_output=True, text=True, encoding="utf-8",
            env=_utf8_env, timeout=120,
        )
    finally:
        if os.path.exists(marker):
            os.remove(marker)
    assert r.returncode == 1, (
        f"eval must exit 1 on a dirty tree, got {r.returncode}: "
        + (r.stdout + r.stderr)[-500:]
    )
    assert "preflight FAIL" in (r.stdout + r.stderr)


@needs_real_assets
@pytest.mark.parametrize("mutate,expect", [
    (lambda p: p["entries"][0].update(world_uid="w999_never_admitted"),
     "not in the admission-manifest eval pool"),
    (lambda p: p["entries"][0].update(manifest_entry_hash="0" * 64),
     "manifest_entry_hash mismatch"),
    (lambda p: p["entries"][0].update(
        role=next(
            r for r in (
                "anticipatory", "dense_recovery", "support_expansion",
                "yaw_lateral_recovery",
            ) if r != p["entries"][0]["role"]
        )),
     # R7-8 ruling (2026-08-24): the eval pool is rl_train ∪ rl_val ∪
     # rl_test — the CLI rejection text carries the canonical term.
     "rl_train ∪ rl_val ∪ rl_test manifest entry"),
])
def test_eval_official_bank_entry_rejected(mutate, expect, _utf8_env,
                                           tmp_path):
    """A bank entry that passes structure but violates the pool/hash/
    manifest contract is rejected at CLI level (exit 2)."""
    payload = _real_bank_payload()
    mutate(payload)
    # Re-sign after the mutation: a byte change breaks the bank_id
    # binding (rejected by the CLI's bank gate before any pool/manifest
    # check). Re-signing simulates the real scenario the semantic gates
    # guard — a bank produced against a DIFFERENT manifest/world set.
    # Mirrors paired_bank_to_dict: strip the injected per-entry ids,
    # hash the remaining content (only bank_id is excluded — bank_version
    # is part of the bound content), re-inject.
    for e in payload["entries"]:
        e.pop("bank_id", None)
        e.pop("bank_version", None)
    payload["bank_id"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "bank_id"}
    )
    for e in payload["entries"]:
        e["bank_id"] = payload["bank_id"]
        e["bank_version"] = "3"
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(payload))
    # A VALID density — the mutated-bank test targets the pool/hash/
    # manifest semantics, and the density load runs before them.
    density = tmp_path / "density.json"
    density.write_text(json.dumps(_real_density_payload()))
    r = subprocess.run(
        [sys.executable, EVAL_OFFICIAL,
         "--checkpoint", str(tmp_path / "none.pt"),
         "--paired_bank", str(bank),
         "--offline_density", str(density),
         "--output_dir", str(tmp_path / "out")],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=120,
    )
    assert r.returncode == 2, (
        f"expected exit 2, got {r.returncode}: "
        + (r.stdout + r.stderr)[-500:]
    )
    assert expect in (r.stdout + r.stderr), (
        f"expected {expect!r}, got: " + (r.stdout + r.stderr)[-500:]
    )


@needs_real_assets
def test_eval_official_identity_mismatch(_utf8_env, tmp_path):
    """A structurally valid checkpoint whose identity diverges from the
    eval-side runtime is rejected before any state load (exit 1).

    Active only on a clean tree — the preflight hard-fails a dirty tree
    before the identity check is reachable.
    """
    if not _preflight_clean():
        pytest.skip("identity check needs a clean tree (post-commit)")
    ckpt = tmp_path / "fake.pt"
    _fake_checkpoint(ckpt)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(_real_bank_payload()))
    density = tmp_path / "density.json"
    density.write_text(json.dumps(_real_density_payload()))
    r = subprocess.run(
        [sys.executable, EVAL_OFFICIAL,
         "--checkpoint", str(ckpt),
         "--paired_bank", str(bank),
         "--offline_density", str(density),
         "--output_dir", str(tmp_path / "out")],
        capture_output=True, text=True, encoding="utf-8",
        env=_utf8_env, timeout=300,
    )
    assert r.returncode == 1, (
        f"identity mismatch must exit 1, got {r.returncode}: "
        + (r.stdout + r.stderr)[-500:]
    )
    assert "checkpoint identity FAIL" in (r.stdout + r.stderr)


def test_official_preflight_passes_on_clean_tree():
    """Positive preflight — the entries' own gate passes on a clean tree.

    Active only post-commit (a dirty tree hard-fails by design). Runs the
    preflight in a subprocess so this test process's torch backend state
    (formal numerical mode) is not mutated.
    """
    if not _preflight_clean():
        pytest.skip("positive preflight needs a clean tree (post-commit)")
    if not os.path.isfile(REAL_BC_CKPT):
        pytest.skip("BC checkpoint not present")
    code = (
        "from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import "
        "validate_official_environment;"
        "f = validate_official_environment("
        f"manifest_path={REAL_MANIFEST!r}, "
        f"bc_checkpoint_path={REAL_BC_CKPT!r}, "
        f"worlds_train_dir={REAL_WORLDS_TRAIN!r});"
        "print(f['project_git_commit'], f['project_tree_hash'], "
        "f['manifest_hash'], f['bc_checkpoint_hash'])"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        encoding="utf-8", timeout=300,
    )
    assert r.returncode == 0, (
        "preflight failed on a clean tree: " + (r.stdout + r.stderr)[-800:]
    )
    parts = r.stdout.split()
    assert len(parts) == 4
    assert len(parts[0]) == 40 and len(parts[1]) == 40
    assert len(parts[2]) == 64 and len(parts[3]) == 64


# ── pure validation functions ───────────────────────────────────────

def test_official_source_file_hashes_cover_core_files():
    """The code-hash axis covers the whole tracked tree, deterministically.

    ``git ls-files`` is the source set — the canonical entry scripts are
    tracked and bound by the identity axis like every other file (a
    moved/deleted script would vanish from the hash set).
    """
    hashes = official_source_file_hashes()
    assert hashes, "source_file_hashes must not be empty"
    for key in (
        "diffusion_policy_3d/rl/dppo_uav/checkpoint.py",
        "diffusion_policy_3d/rl/dppo_uav/train_loop.py",
        "diffusion_policy_3d/rl/dppo_uav/entrypoint_guard.py",
        "diffusion_policy_3d/rl/dppo_uav/capability_metrics.py",
        "scripts/train_dppo_uav_official.py",
        "scripts/eval_dppo_uav_official.py",
    ):
        assert key in hashes, f"missing from source hashes: {key}"
    assert all(len(v) == 64 and all(c in "0123456789abcdef" for c in v)
               for v in hashes.values()), "non-sha256 hash value"
    assert official_source_file_hashes() == hashes, (
        "source hashes must be deterministic"
    )


def test_official_source_file_hash_matches_direct_sha256():
    """One entry verified against a direct sha256 of the file bytes."""
    hashes = official_source_file_hashes()
    path = "diffusion_policy_3d/rl/dppo_uav/entrypoint_guard.py"
    direct = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert hashes[path] == direct


def test_assert_no_eval_only_worlds_pure():
    """The eval-only leak check is a pure function over the pool uids."""
    assert_no_eval_only_worlds(["w1", "w2", "w3"])
    with pytest.raises(RuntimeError, match="eval-only"):
        assert_no_eval_only_worlds(["w1", "w4", "w2"])
    assert "w4" in EVAL_ONLY_WORLDS and "w18" in EVAL_ONLY_WORLDS


def test_vendor_hashes_verify():
    """The frozen vendor snapshot (401 files) passes sha256sum -c."""
    try:
        verify_vendor_hashes()
    except RuntimeError as e:
        pytest.fail(f"vendor hash verification failed: {e}")



def test_g7_micro_stage_registered_in_canonical_entry():
    """G7 (design §5.9: 30-60 PPO updates) needs a canonical-entry stage
    inside that window — the entry's own stages (micro_train=10,
    pilot=100) bracket it without covering it. The g7_micro stage fixes
    the pre-registered budget at 30 updates, one_update mode."""
    from scripts.train_dppo_uav_official import STAGE_CONFIG

    assert "g7_micro" in STAGE_CONFIG, (
        "g7_micro stage missing — the G7 pre-registration budget "
        "(30-60 updates per design §5.9) has no canonical entry stage"
    )
    mode, budget = STAGE_CONFIG["g7_micro"]
    assert mode == "one_update", mode
    assert 30 <= budget <= 60, budget
