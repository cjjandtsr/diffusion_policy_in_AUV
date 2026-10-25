"""R6-D (R6-EVAL-2): capability opportunity schema and the five-piece
outcome contract (F-04).

Guide §11.5.3/§11.5.5:

- opportunity fields are frozen, policy-independent facts (nominal path
  geometry / world topology / offline density) — an opportunity whose
  provenance is the evaluated trajectory is REJECTED, never silently
  accepted;
- every capability group outputs the five pieces
  {opportunity_count, exposure_count, event_count, value|null,
   validity_reason};
- opportunity_count == 0 => NOT_INFORMATIVE, never a 0 score;
- missing opportunity on a v3 bank entry is NOT silently defaulted —
  it stays None and the gate reports NOT_INFORMATIVE;
- opportunity>0 but exposure=0: exposure is explicitly retained and the
  value is handled per its preregistered definition;
- the bank round-trip carries opportunity and rejects invalid blocks.
"""

import math

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    OfflineDensity,
    CapabilityOpportunity,
    capability_informativeness,
    compute_capability_outcomes,
    opportunity_from_dict,
    opportunity_to_dict,
    validate_opportunity,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSpec,
    canonical_sha256,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    CAPABILITY_METRICS,
    EpisodePairedRecord,
    _bank_content,
    freeze_paired_bank,
    paired_bank_from_dict,
    paired_bank_to_dict,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.paired_rng import (
    RNG_CONTRACT_VERSION,
    policy_noise_seed,
)

from mock_uav_env import MockUAVEnv

START = [-8.0, 0.0, 2.0]
GOAL = [8.0, 0.0, 2.0]

FIVE_PIECE_KEYS = ("opportunity_count", "exposure_count", "event_count",
                   "value", "validity_reason")


def _step(i: int, pos: list, *, clearance: float, yaw: float = 0.0,
          terminated: bool = False, done_reason: str = "none") -> dict:
    return {
        "step": i, "elapsed": 0.1 * i,
        "pos": [float(v) for v in pos], "z": float(pos[2]),
        "yaw": float(yaw),
        "goal_dist": float(math.dist(pos, GOAL)),
        "clearance": float(clearance), "hit_count": 0,
        "action_norm": 1.0, "action_max_abs": 0.5,
        "done_reason": done_reason,
        "terminated": bool(terminated), "truncated": False,
    }


def _straight(n: int, *, y: float = 0.0, z: float = 2.0,
              clearances: list = None) -> list:
    cl = (clearances or [4.0] * n)
    assert len(cl) == n
    return [
        _step(i, [-8.0 + i, y, z], clearance=cl[i]) for i in range(n)
    ]


def _density() -> OfflineDensity:
    """Corridor (x_bin 0..16, y_bin 0, z_bin 4) visited 100x; periphery
    cells (8, +-2, 4) visited once. p20 = 100 => corridor is NOT
    low-density, everything else IS (the test_capability_metrics
    fixture)."""
    counts = {(x, 0, 4): 100 for x in range(17)}
    counts[(8, 2, 4)] = 1
    counts[(8, -2, 4)] = 1
    counts[(9, 2, 4)] = 1
    counts[(9, -2, 4)] = 1
    return OfflineDensity(counts=counts)


def _record(role: str, bc_trace: list, rl_trace: list, *,
            bc_success: bool, rl_success: bool,
            bc_collision: bool = False, rl_collision: bool = False,
            ep_hash: str = "ep_0000", seed: int = 42) -> EpisodePairedRecord:
    nseed = policy_noise_seed("bank-abc", ep_hash, 0)
    return EpisodePairedRecord(
        episode_config={"world": "w1", "seed": seed,
                        "episode_config_hash": ep_hash},
        seed=seed, world="w1", role=role,
        start=list(START), goal=list(GOAL),
        disturbance={"events": []},
        bc={"success": bc_success, "collision": bc_collision,
            "timeout": False, "return": 10.0, "steps": len(bc_trace)},
        rl={"success": rl_success, "collision": rl_collision,
            "timeout": False, "return": 10.0, "steps": len(rl_trace)},
        trace={"bc": bc_trace, "rl": rl_trace},
        policy_noise_seed=nseed, replica_id=0,
        rng_contract_version=RNG_CONTRACT_VERSION,
    )


def _opportunity(**overrides) -> CapabilityOpportunity:
    base = dict(
        valid=True,
        nominal_path_risk=0.1,
        risk_shell_crossing=True,
        anticipatory_provenance="nominal_path_geometry",
        topology_class="trap",
        trap_opportunity_count=1,
        dense_provenance="world_topology",
        support_stratum="outside",
        reference_density=0.0,
        support_provenance="offline_density",
    )
    base.update(overrides)
    return CapabilityOpportunity(**base)


def _opp_map(records, op_factory=_opportunity) -> dict:
    """One opportunity block per record's episode_config_hash."""
    return {r.episode_config["episode_config_hash"]: op_factory()
            for r in records}


# ═══════════════════════════════════════════════════════════════════
# opportunity schema: construction, rejection, policy-independence
# ═══════════════════════════════════════════════════════════════════

def test_opportunity_construction_defaults_and_roundtrip():
    op = _opportunity()
    assert op.valid and op.risk_shell_crossing
    assert op.trap_opportunity_count == 1
    assert op.support_stratum == "outside"
    # Minimal block: only the fields that are meaningful for the entry.
    minimal = CapabilityOpportunity(
        valid=True,
        anticipatory_provenance="nominal_path_geometry",
    )
    assert minimal.risk_shell_crossing is False
    assert minimal.trap_opportunity_count == 0
    assert minimal.support_stratum == ""
    # dict round-trip is lossless.
    assert opportunity_from_dict(opportunity_to_dict(op)) == op


def test_validate_opportunity_rejections():
    with pytest.raises(ValueError, match="bool"):
        _opportunity(trap_opportunity_count=True)
    with pytest.raises(ValueError, match="non-negative"):
        _opportunity(trap_opportunity_count=-1)
    with pytest.raises(ValueError, match="finite"):
        _opportunity(nominal_path_risk=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _opportunity(reference_density=float("inf"))
    with pytest.raises(ValueError, match="topology_class"):
        _opportunity(topology_class="corridor")
    with pytest.raises(ValueError, match="support_stratum"):
        _opportunity(support_stratum="inside2")
    # provenance is REQUIRED for every group that declares a field.
    with pytest.raises(ValueError, match="provenance"):
        _opportunity(anticipatory_provenance="")
    with pytest.raises(ValueError, match="provenance"):
        _opportunity(anticipatory_provenance="from_manual_inspection")
    validate_opportunity(_opportunity())  # the valid block passes


def test_policy_dependent_opportunity_rejected():
    """§11.5.3: opportunity must come from frozen geometry/nominal
    path/offline density — an opportunity derived from the EVALUATED
    policy's trajectory is rejected (the F-04 counterexample)."""
    with pytest.raises(ValueError, match="policy-dependent"):
        _opportunity(anticipatory_provenance="evaluated_trace")
    with pytest.raises(ValueError, match="policy-dependent"):
        _opportunity(dense_provenance="evaluated_trace")
    with pytest.raises(ValueError, match="policy-dependent"):
        _opportunity(support_provenance="evaluated_trace")


def test_bank_roundtrip_opportunity_absent_none():
    """The v3 bank stays readable WITHOUT opportunity; a capability bank
    carries it; invalid blocks are rejected — never silently defaulted."""
    contract = RLSystemContract()
    entries = freeze_paired_bank(
        MockUAVEnv(contract),
        [
            EpisodeTaskSpec(
                world_uid="mock_anticipatory", obstacle_json="{}",
                role="anticipatory", split="rl_train",
                admission_id="mock__anticipatory",
                generator_version="mock",
                manifest_entry_hash="hash_anticipatory"),
        ],
        [1001],
    )
    assert entries[0].opportunity is None  # frozen v3 has no opportunity
    bank = paired_bank_to_dict(entries, "src", "abc")
    read = paired_bank_from_dict(bank)
    assert read[0].opportunity is None
    # Inject a valid opportunity block, rebind, reload.
    bank["entries"][0]["opportunity"] = opportunity_to_dict(_opportunity())
    bank["bank_id"] = canonical_sha256(_bank_content(bank))
    read2 = paired_bank_from_dict(bank)
    assert read2[0].opportunity == _opportunity()
    # Inject an INVALID block (policy-dependent provenance): rejected.
    # The opportunity constructor refuses invalid blocks, so the bad
    # block is built by tampering a valid one — the bank-file scenario
    # (a hand-written or corrupted bank entry).
    bad_block = opportunity_to_dict(_opportunity())
    bad_block["anticipatory_provenance"] = "evaluated_trace"
    bank["entries"][0]["opportunity"] = bad_block
    bank["bank_id"] = canonical_sha256(_bank_content(bank))
    with pytest.raises(ValueError, match="policy-dependent"):
        paired_bank_from_dict(bank)


# ═══════════════════════════════════════════════════════════════════
# the five-piece contract
# ═══════════════════════════════════════════════════════════════════

def test_five_piece_schema_all_groups():
    records = [
        _record("anticipatory", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0000"),
        _record("support_expansion", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0001"),
        _record("dense_recovery", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0002"),
    ]
    out = compute_capability_outcomes(records, _opp_map(records),
                                      offline_density=_density())
    assert set(out) == set(CAPABILITY_METRICS)
    for group, pieces in out.items():
        assert set(pieces) == set(FIVE_PIECE_KEYS), group
        assert pieces["opportunity_count"] == 1
        assert isinstance(pieces["exposure_count"], int)
        assert isinstance(pieces["event_count"], int)
        assert "validity_reason" in pieces


def test_anticipatory_informative_value_hand_crafted():
    """RL onset 4 steps before the shell (clearance 3.0 at onset), BC
    onset 2 steps before (clearance 1.5): paired value = 3.0 - 1.5."""
    clearances = [4.0] * 4 + [3.0, 2.0, 1.5, 0.8, 0.5, 0.5]
    bc_trace = _straight(10, clearances=clearances)
    rl_trace = _straight(10, clearances=clearances)
    for i in range(6, 10):
        bc_trace[i]["pos"][1] = 0.5
    for i in range(4, 10):
        rl_trace[i]["pos"][1] = 0.5
    record = _record("anticipatory", bc_trace, rl_trace,
                     bc_success=True, rl_success=True)
    out = compute_capability_outcomes([record], _opp_map([record]),
                                      offline_density=_density())
    pieces = out["anticipatory_avoidance"]
    assert pieces["opportunity_count"] == 1
    assert pieces["exposure_count"] == 1   # both sides enter the shell
    assert pieces["event_count"] == 1      # both sides show an onset
    assert pieces["value"] == pytest.approx(3.0 - 1.5)
    assert pieces["validity_reason"] is None


def test_opportunity_zero_is_not_informative_never_a_score():
    """§11.5.3: opportunity_count=0 => NOT_INFORMATIVE; value is null,
    NOT 0 — a 0 score would claim evidence where the bank has none."""
    records = [
        _record("anticipatory", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0000"),
    ]
    no_shell_opp = _opp_map(records,
                            op_factory=lambda: _opportunity(
                                risk_shell_crossing=False))
    out = compute_capability_outcomes(records, no_shell_opp,
                                      offline_density=_density())
    pieces = out["anticipatory_avoidance"]
    assert pieces["opportunity_count"] == 0
    assert pieces["value"] is None            # not 0.0
    assert pieces["validity_reason"] is not None
    status = capability_informativeness(out)
    assert status["anticipatory_avoidance"]["status"] == "NOT_INFORMATIVE"
    assert "opportunity" in status["anticipatory_avoidance"]["reason"]


def test_missing_opportunity_not_silently_defaulted():
    """§11.5.3: a bank without opportunity blocks stays readable but the
    gate reports NOT_INFORMATIVE — missing is never treated as
    opportunity=0 with a score."""
    records = [
        _record("anticipatory", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0000"),
    ]
    out = compute_capability_outcomes(records, {}, offline_density=_density())
    pieces = out["anticipatory_avoidance"]
    assert pieces["opportunity_count"] == 0
    assert pieces["value"] is None
    status = capability_informativeness(out)
    assert status["anticipatory_avoidance"]["status"] == "NOT_INFORMATIVE"
    assert "absent" in status["anticipatory_avoidance"]["reason"]


def test_exposure_zero_explicitly_retained():
    """§11.5.5: opportunity>0 but exposure=0 — the exposure count stays
    0 (explicit), value follows its preregistered definition (null here,
    no onset episodes), and the gate stays INFORMATIVE (opportunity
    exists; the zero is reported, not hidden)."""
    clearances = [4.0] * 10  # never enters the shell
    record = _record("anticipatory", _straight(10, clearances=clearances),
                     _straight(10, clearances=clearances),
                     bc_success=True, rl_success=True)
    out = compute_capability_outcomes([record], _opp_map([record]),
                                      offline_density=_density())
    pieces = out["anticipatory_avoidance"]
    assert pieces["opportunity_count"] == 1
    assert pieces["exposure_count"] == 0
    assert pieces["event_count"] == 0
    assert pieces["value"] is None
    assert "exposure" in pieces["validity_reason"]
    status = capability_informativeness(out)
    assert status["anticipatory_avoidance"]["status"] == "INFORMATIVE"


def test_dense_no_trap_opportunity_zero_null():
    """§11.5.5: trap_segments=0 and opportunity=0 => null /
    NOT_INFORMATIVE for the dense group."""
    records = [
        _record("dense_recovery", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0002"),
    ]
    no_trap = _opp_map(records,
                       op_factory=lambda: _opportunity(
                           trap_opportunity_count=0))
    out = compute_capability_outcomes(records, no_trap,
                                      offline_density=_density())
    pieces = out["dense_obstacle_ceiling"]
    assert pieces["opportunity_count"] == 0
    assert pieces["value"] is None
    assert pieces["validity_reason"] is not None
    status = capability_informativeness(out)
    assert status["dense_obstacle_ceiling"]["status"] == "NOT_INFORMATIVE"


def test_dense_informative_recovery_rate_delta():
    """BC: one open trap segment [13..16], never recovered (rate 0.0);
    RL: one 3-step segment [13..15], recovered (rate 1.0) — paired value
    1.0 - 0.0 = 1.0."""
    def _dense_bc():
        trace = []
        for i in range(9):
            trace.append(_step(i, [-8.0 + i, 0.0, 2.0], clearance=4.0))
        y_side = [0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4]
        cl = [0.7, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3]
        for k, (yy, cc) in enumerate(zip(y_side, cl)):
            i = 9 + k
            trace.append(_step(
                i, [0.0, yy, 2.0], clearance=cc,
                terminated=(i == 16), done_reason=("collision" if i == 16
                                                   else "none")))
        return trace

    def _dense_rl():
        trace = []
        for i in range(9):
            trace.append(_step(i, [-8.0 + i, 0.0, 2.0], clearance=4.0))
        y_seq = [0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4]
        cl = [0.7] + [0.5] * 7
        for k in range(7):
            trace.append(_step(9 + k, [0.0, y_seq[k], 2.0], clearance=cl[k]))
        for k, xx in enumerate([2.0, 4.0, 6.0, 8.0]):
            i = 16 + k
            trace.append(_step(
                i, [xx, 0.4, 2.0], clearance=[1.0, 1.8, 2.6, 3.4][k],
                terminated=(i == 19), done_reason=("reach" if i == 19
                                                   else "none")))
        return trace

    record = _record("dense_recovery", _dense_bc(), _dense_rl(),
                     bc_success=False, bc_collision=True, rl_success=True)
    out = compute_capability_outcomes([record], _opp_map([record]),
                                      offline_density=_density())
    pieces = out["dense_obstacle_ceiling"]
    assert pieces["opportunity_count"] == 1
    assert pieces["exposure_count"] == 1   # obstacle interactions both sides
    assert pieces["event_count"] == 1      # trap segments both sides
    assert pieces["value"] == pytest.approx(1.0 - 0.0)
    assert pieces["validity_reason"] is None


def test_support_informative_out_of_support_share_delta():
    """BC stays in the corridor (share 0); RL flies at z=2.5 — the whole
    trace is outside BC support (share 1.0): value = 0 - 1.0."""
    bc_trace = _straight(10)
    rl_trace = _straight(10, z=2.5)
    record = _record("support_expansion", bc_trace, rl_trace,
                     bc_success=True, rl_success=True)
    out = compute_capability_outcomes([record], _opp_map([record]),
                                      offline_density=_density())
    pieces = out["fullspace_support_expansion"]
    assert pieces["opportunity_count"] == 1
    assert pieces["exposure_count"] == 1
    assert pieces["event_count"] == 1
    assert pieces["value"] == pytest.approx(0.0 - 1.0)
    assert pieces["validity_reason"] is None


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-EVAL-2: CapabilityOpportunity.valid 必须生效
# ═══════════════════════════════════════════════════════════════════

def test_invalid_opportunity_not_counted_not_informative():
    """Gate B R6-EVAL-2: valid=False 的 block 即使字段声明 opportunity
    （risk_shell_crossing=True），也不得计入 opportunity_count ——
    valid=False -> opportunity_count=0 -> NOT_INFORMATIVE，原因显式
    区分 no_valid_opportunity（"present but invalid"），不是静默的
    no_opportunity。"""
    records = [
        _record("anticipatory", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0000"),
    ]
    invalid = _opp_map(records, op_factory=lambda: _opportunity(valid=False))
    out = compute_capability_outcomes(records, invalid,
                                      offline_density=_density())
    pieces = out["anticipatory_avoidance"]
    assert pieces["opportunity_count"] == 0
    assert pieces["value"] is None            # 不是 0 分
    assert "valid" in pieces["validity_reason"]
    status = capability_informativeness(out)
    assert status["anticipatory_avoidance"]["status"] == "NOT_INFORMATIVE"
    assert "valid" in status["anticipatory_avoidance"]["reason"]


def test_mixed_validity_only_valid_opportunities_counted():
    """valid=True 的 entry 正常计数，valid=False 的 entry 被排除。"""
    rec_a = _record("anticipatory", _straight(10), _straight(10),
                    bc_success=True, rl_success=True, ep_hash="ep_0000")
    rec_b = _record("anticipatory", _straight(10), _straight(10),
                    bc_success=True, rl_success=True, ep_hash="ep_0001")
    opp = _opp_map([rec_a, rec_b])
    opp[rec_b.episode_config["episode_config_hash"]] = _opportunity(
        valid=False)
    out = compute_capability_outcomes([rec_a, rec_b], opp,
                                      offline_density=_density())
    pieces = out["anticipatory_avoidance"]
    assert pieces["opportunity_count"] == 1
    status = capability_informativeness(out)
    assert status["anticipatory_avoidance"]["status"] == "INFORMATIVE"


def test_invalid_dense_and_support_not_counted():
    """valid 生效覆盖全部三组：dense 的 trap_opportunity_count 与
    support 的 outside stratum 同样被 valid=False 抑制。"""
    recs = [
        _record("dense_recovery", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0002"),
        _record("support_expansion", _straight(10), _straight(10),
                bc_success=True, rl_success=True, ep_hash="ep_0003"),
    ]
    opp = _opp_map(recs, op_factory=lambda: _opportunity(valid=False))
    out = compute_capability_outcomes(recs, opp, offline_density=_density())
    assert out["dense_obstacle_ceiling"]["opportunity_count"] == 0
    assert out["fullspace_support_expansion"]["opportunity_count"] == 0
    status = capability_informativeness(out)
    assert status["dense_obstacle_ceiling"]["status"] == "NOT_INFORMATIVE"
    assert status["fullspace_support_expansion"]["status"] == "NOT_INFORMATIVE"
