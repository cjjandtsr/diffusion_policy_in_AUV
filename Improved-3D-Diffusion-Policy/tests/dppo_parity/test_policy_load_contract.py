"""R6-E §11.6.1 (F-06 → R6-LOAD-1): checkpoint weight-load contract.

validate_policy_load_contract closes the strict=False gap in the BC loader
(policy_loader.py:139): before LoRA insertion and the first rollout,

    real_missing == []
    set(unexpected) == exact_allowlist_for_this_bc_hash

The allowlist is bound to the REGISTERED BC checkpoint SHA256 + the
loader/schema version. R7-1 (2026-08-22, user ruling): the registered
base actor is e5_lambda0p5 (E5 retrain, V11-fixed collision loss) —
measured state_dict = 192 keys, ZERO anchor keys (cfg.policy.
anchor_bank_path=None), so the allowlist is EMPTY and the strict=False
load is provably total (missing=[] unexpected=[]). Schema bumped v1→v2
for the head-layout change (7 anchor keys → 0).

The negative classes — a missing UNet key, ANY unexpected key (with an
empty allowlist every unexpected key is extra), and a wrong BC hash —
must all hard-fail BEFORE any env construction/rollout.

Run: pytest -q tests/dppo_parity/test_policy_load_contract.py
"""

import dataclasses

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    EvalArtifactError,
    validate_eval_artifact,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
    BC_ANCHOR_ALLOWLIST,
    BC_COMPATIBILITY_MANIFEST,
    LOAD_CONTRACT_SCHEMA_VERSION,
    REGISTERED_BC_CHECKPOINT_HASH,
    PolicyLoadAudit,
    validate_policy_load_contract,
)

# The registered BC checkpoint (R7/G8 base actor e5_lambda0p5):
# data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt —
# sha256 of the file bytes (measured 2026-08-22).
BC_SHA = REGISTERED_BC_CHECKPOINT_HASH

# e5_lambda0p5 carries NO anchor keys (192 keys measured 2026-08-22) —
# the runtime policy constructed from its own cfg.policy loads with
# zero unexpected keys, so the allowlist is empty.
ANCHOR_KEYS = ()


def _valid_manifest() -> dict:
    return dict(BC_COMPATIBILITY_MANIFEST)


def _valid_call(**over) -> dict:
    call = dict(
        bc_checkpoint_hash=BC_SHA,
        missing_keys=[],
        unexpected_keys=list(ANCHOR_KEYS),
        compatibility_manifest=_valid_manifest(),
    )
    call.update(over)
    return call


# ───────────────────────────────────────────────────────────────────
# §11.6.1 positive: the exact contract passes and returns the audit
# ───────────────────────────────────────────────────────────────────

def test_audit_passes_exact_allowlist():
    audit = validate_policy_load_contract(**_valid_call())
    assert isinstance(audit, PolicyLoadAudit)
    assert audit.missing_keys == ()
    assert set(audit.unexpected_keys) == set(ANCHOR_KEYS)
    assert audit.allowlisted_keys == ANCHOR_KEYS
    assert audit.bc_checkpoint_hash == BC_SHA
    assert audit.loader_schema_version == LOAD_CONTRACT_SCHEMA_VERSION
    assert dict(audit.compatibility_manifest) == dict(BC_COMPATIBILITY_MANIFEST)


# ───────────────────────────────────────────────────────────────────
# §11.6.1 negative class 1: a missing UNet key
# ───────────────────────────────────────────────────────────────────

def test_missing_unet_key_hard_fails():
    with pytest.raises(ValueError, match="missing"):
        validate_policy_load_contract(
            **_valid_call(missing_keys=["unet.0.weight"])
        )


# ───────────────────────────────────────────────────────────────────
# §11.6.1 negative class 2: an unexpected key NOT in the allowlist
# ───────────────────────────────────────────────────────────────────

def test_unexpected_key_not_in_allowlist_hard_fails():
    with pytest.raises(ValueError, match="allowlist"):
        validate_policy_load_contract(
            **_valid_call(unexpected_keys=list(ANCHOR_KEYS)
                          + ["unet.5.weight"])
        )


# ───────────────────────────────────────────────────────────────────
# §11.6.1 negative class 3: wrong BC hash
# ───────────────────────────────────────────────────────────────────

def test_wrong_bc_hash_hard_fails():
    with pytest.raises(ValueError, match="[Uu]nregistered|BC checkpoint"):
        validate_policy_load_contract(
            **_valid_call(bc_checkpoint_hash="00" * 32)
        )


# ───────────────────────────────────────────────────────────────────
# §11.6.1 negative class 4: an anchor-SHAPED key under schema v2
# (the old 6/7-anchors-present guard is superseded: e5 has no anchors,
# so the empty exact-set allowlist tolerates nothing)
# ───────────────────────────────────────────────────────────────────

def test_anchor_shaped_key_hard_fails_under_schema_v2():
    """Schema v2 (2026-08-22 user ruling): e5_lambda0p5 measured 192
    keys / ZERO anchor keys — an anchor-shaped unexpected key is EXTRA,
    not tolerated. The exact-set allowlist is empty, so any non-empty
    unexpected set fails (no 'may be present or absent')."""
    with pytest.raises(ValueError, match="allowlist"):
        validate_policy_load_contract(
            **_valid_call(unexpected_keys=["anchor_head.net.4.bias"])
        )


# ───────────────────────────────────────────────────────────────────
# allowlist shape: EMPTY literal set under schema v2, no wildcards,
# bound to the measured e5 state_dict (192 keys, 0 anchor keys)
# ───────────────────────────────────────────────────────────────────

def test_allowlist_is_empty_and_wildcard_free():
    """Schema v2 (2026-08-22 user ruling): the registered base actor
    e5_lambda0p5 carries zero anchor keys, so the allowlist is EMPTY —
    and any future edit must stay literal (no wildcards) and stay bound
    to ANCHOR_KEYS (the measured unexpected set, also empty)."""
    assert BC_ANCHOR_ALLOWLIST == ()
    assert len(set(BC_ANCHOR_ALLOWLIST)) == 0
    for key in BC_ANCHOR_ALLOWLIST:
        assert "*" not in key, f"wildcard forbidden: {key}"
        assert "?" not in key, f"wildcard forbidden: {key}"
    assert BC_ANCHOR_ALLOWLIST == ANCHOR_KEYS


def test_manifest_explains_empty_allowlist():
    """With ZERO allowlisted keys the manifest must still: (a) record
    the loader schema version, (b) explain WHY the allowlist is empty
    (the e5 adoption ruling), and (c) keep excluded_keys exactly aligned
    with the allowlist — both empty."""
    excluded = BC_COMPATIBILITY_MANIFEST["excluded_keys"]
    assert excluded == {}
    assert set(BC_ANCHOR_ALLOWLIST) == set(excluded), (
        "manifest must explain exactly the allowlisted keys"
    )
    assert BC_COMPATIBILITY_MANIFEST["loader_schema_version"] == (
        LOAD_CONTRACT_SCHEMA_VERSION
    )
    assert BC_COMPATIBILITY_MANIFEST["summary"].strip()
    assert "e5_lambda0p5" in BC_COMPATIBILITY_MANIFEST["summary"], (
        "manifest summary must identify the registered base actor"
    )


def test_manifest_drift_hard_fails():
    """A mutated manifest (e.g. someone silently ADDING an exclusion or
    editing the summary) must not load under the frozen contract — the
    manifest is a frozen constant, not a per-run explanation."""
    drifted = dict(BC_COMPATIBILITY_MANIFEST)
    drifted["excluded_keys"] = dict(drifted["excluded_keys"])
    drifted["excluded_keys"]["unet.5.weight"] = "silent addition"
    with pytest.raises(ValueError, match="manifest"):
        validate_policy_load_contract(**_valid_call(
            compatibility_manifest=drifted
        ))


# ───────────────────────────────────────────────────────────────────
# artifact side: the recorded unexpected keys must self-explain via the
# load contract (§11.6.1 "被纳入 … artifact")
# ───────────────────────────────────────────────────────────────────

def _artifact_with(**over) -> object:
    """Shared §4.17 artifact fixture (test_eval_artifact._artifact) with
    the unexpected-key / load-contract knobs overridden."""
    from test_eval_artifact import _artifact

    return dataclasses.replace(_artifact(), **over)


def _valid_load_contract() -> dict:
    return {
        "bc_checkpoint_hash": BC_SHA,
        "loader_schema_version": LOAD_CONTRACT_SCHEMA_VERSION,
        "compatibility_manifest": _valid_manifest(),
    }


def test_artifact_unexpected_keys_require_load_contract():
    """ANY recorded unexpected key (under schema v2 every unexpected key
    is extra) without the §11.6.1 load_contract cannot self-explain —
    hard fail before the key set is even compared."""
    a = _artifact_with(unexpected_weight_keys=["unet.5.weight"])
    with pytest.raises(EvalArtifactError, match="load_contract"):
        validate_eval_artifact(a)


def test_artifact_load_contract_unregistered_hash_hard_fails():
    a = _artifact_with(
        unexpected_weight_keys=["unet.5.weight"],
        load_contract=dict(_valid_load_contract(),
                           bc_checkpoint_hash="00" * 32),
    )
    with pytest.raises(EvalArtifactError, match="BC checkpoint|unregistered"):
        validate_eval_artifact(a)


def test_artifact_load_contract_unexpected_superset_hard_fails():
    """A valid contract cannot legitimize ANY unexpected key under v2:
    the empty allowlist is exact, so 'unet.5.weight' is a superset."""
    a = _artifact_with(
        unexpected_weight_keys=["unet.5.weight"],
        load_contract=_valid_load_contract(),
    )
    with pytest.raises(EvalArtifactError, match="allowlist"):
        validate_eval_artifact(a)


def test_artifact_v1_anchor_keys_rejected_under_v2_schema():
    """Schema v2 (empty exact allowlist): a v1-era artifact recording an
    anchor unexpected key is rejected even WITH a plausible contract —
    the allowlist check fires before the manifest is consulted, so
    manifest drift is unreachable through the artifact path in v2 (the
    loader-level test_manifest_drift_hard_fails covers the manifest)."""
    a = _artifact_with(
        unexpected_weight_keys=["anchor_head.net.4.bias"],
        load_contract=_valid_load_contract(),
    )
    with pytest.raises(EvalArtifactError, match="allowlist"):
        validate_eval_artifact(a)


def test_artifact_load_contract_valid_passes_and_round_trips():
    """The frozen load contract is carried by the artifact (to_dict /
    from_dict) and validates — with zero unexpected keys the contract is
    provenance (recorded at load time), not a key-legitimizer."""
    a = _artifact_with(
        unexpected_weight_keys=[],
        load_contract=_valid_load_contract(),
    )
    validate_eval_artifact(a)
    import json
    d = json.loads(json.dumps(a.to_dict()))
    assert d["load_contract"]["bc_checkpoint_hash"] == BC_SHA
    assert d["load_contract"]["loader_schema_version"] == (
        LOAD_CONTRACT_SCHEMA_VERSION
    )
    assert d["load_contract"]["compatibility_manifest"] == (
        _valid_manifest()
    )


def test_artifact_no_unexpected_keys_without_contract_passes():
    """A BC checkpoint with zero unexpected keys needs no manifest — the
    contract is bound to the anchor-carrying registered checkpoint."""
    a = _artifact_with(unexpected_weight_keys=[])
    validate_eval_artifact(a)
