"""EVAL-AUDIT (§4.17): episode-level paired eval artifact schema + hard-fail
rules.

§4.17 requires per artifact:
  - episode-level BC/RL paired records (not just aggregates);
  - episode config, seed, world, role, start, goal, disturbance;
  - BC/RL success, collision, timeout, return, steps;
  - three capability metrics;
  - checkpoint/config/contract/manifest/code hash;
  - reward version and exec steps;
  - observation invalid count;
  - missing/unexpected weight keys;
  - LoRA parameter checksum;
  - evaluation timestamp and device.

[JUDGMENT] formal eval with missing LoRA, critical key missing, contract
mismatch or invalid point cloud MUST hard-fail — never warn-and-continue
into a PPO==BC result.
"""

import dataclasses
import json

import pytest

from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    EvalArtifact,
    EpisodePairedRecord,
    EvalArtifactError,
    build_eval_artifact,
    validate_eval_artifact,
)
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    CAPABILITY_METRIC_KEYS,
    FIVE_PIECE_KEYS,
    GROUP_OF_ROLE,
    OfflineDensity,
    OPPORTUNITY_INVALID_REASONS,
    CapabilityOpportunity,
    capability_informativeness,
    compute_capability_metrics,
)


def _step(i: int) -> dict:
    """One valid raw per-step trace entry."""
    return {
        "step": i, "elapsed": 0.1 * i,
        "pos": [float(i), 0.0, 1.0], "z": 1.0, "yaw": 0.0,
        "goal_dist": 5.0, "clearance": 1.0, "hit_count": 4096,
        "action_norm": 1.0, "action_max_abs": 0.5,
        "done_reason": "none", "terminated": False, "truncated": False,
    }


def _trace(n_steps: int = 3) -> dict:
    """The RAW per-step series on both sides — never aggregate."""
    return {"bc": [_step(i) for i in range(n_steps)],
            "rl": [_step(i) for i in range(n_steps)]}


def _paired(role: str = "anticipatory",
            bank_id: str = "bank123",
            ep_hash: str = None) -> EpisodePairedRecord:
    # R5.8-E (N4): the recorded noise seed must re-derive from the
    # artifact's own provenance (bank_id + episode_config_hash +
    # replica) — the shared fixture computes it like the production
    # path does instead of leaving the default 0.
    from diffusion_policy_3d.rl.dppo_uav.paired_rng import policy_noise_seed

    ep_hash = ep_hash if ep_hash is not None else "e0" * 32
    return EpisodePairedRecord(
        episode_config={"world": "w1", "seed": 42,
                        "episode_config_hash": ep_hash},
        seed=42,
        world="w1",
        role=role,
        start=[0.0, 0.0, 0.0],
        goal=[1.0, 2.0, 3.0],
        disturbance={"wind": 0.0},
        # R6-E (§11.6.4): the raw terminated/truncated flags and the
        # mutually exclusive terminal_category ride the record; the
        # artifact validator re-derives the category from the raw flags
        # and hard-fails on mismatch (unmappable states are never
        # silently swallowed). rl side: collision+done_reason="reached"
        # is a CONTRADICTION under the strict mapper — done_reason must
        # be "collision".
        bc={"success": True, "collision": False, "timeout": False, "done_reason": "reached", "out_of_bounds": False, "altitude_exceed": False,
            "terminated": True, "truncated": False, "terminal_category": "reached",
            "return": 95.0, "steps": 40},
        rl={"success": False, "collision": True, "timeout": False, "done_reason": "collision", "out_of_bounds": False, "altitude_exceed": False,
            "terminated": True, "truncated": False, "terminal_category": "collision",
            "return": -20.0, "steps": 12},
        trace=_trace(),
        policy_noise_seed=policy_noise_seed(bank_id, ep_hash, 0),
    )


# One record per capability role — capability metrics are computed per
# role, and the artifact validator requires each role's n_episodes to
# align with the record count.
def _three_role_records(bank_id: str = "bank123") -> list:
    return [
        _paired("anticipatory", bank_id=bank_id),
        _paired("support_expansion", bank_id=bank_id),
        _paired("dense_recovery", bank_id=bank_id),
    ]


def _density() -> OfflineDensity:
    """Corridor field covering the constant-clearance trace cells — a
    low-density cell would change the metrics, not the schema."""
    return OfflineDensity(
        counts={(x, 0, 2): 5 for x in range(17)},
    )


def _caps_for(episodes: list) -> dict:
    """Valid capability schema for an arbitrary episode set — groups
    whose role is absent get n_episodes=0, matching the validator's
    role alignment."""
    caps = {}
    for group, role in GROUP_OF_ROLE.items():
        n = sum(1 for ep in episodes if ep.role == role)
        caps[group] = {
            key: (n if key == "n_episodes" else 0.0)
            for key in CAPABILITY_METRIC_KEYS[group]
        }
    return caps


def _outcomes_for(episodes: list) -> dict:
    """NOT_INFORMATIVE-shaped five pieces (§11.5.3) for an arbitrary
    episode set — opportunity_count=0, value null, frozen validity_reason:
    the honest no-evidence state, never a fabricated 0 score."""
    return {
        group: {
            "opportunity_count": 0,
            "exposure_count": 0,
            "event_count": 0,
            "value": None,
            "validity_reason": OPPORTUNITY_INVALID_REASONS["no_opportunity"],
        }
        for group in CAPABILITY_METRIC_KEYS
    }


def _artifact() -> EvalArtifact:
    records = _three_role_records()
    return build_eval_artifact(
        episodes=records,
        checkpoint_hash="ckpt123",
        config_hash="cfg456",
        contract_hash="contract789",
        manifest_hash="manifestabc",
        world_assets_hash="world123",
        code_hash="codedef",
        reward_version="v16",
        exec_steps=4,
        observation_invalid_count=0,
        missing_weight_keys=[],
        unexpected_weight_keys=[],
        lora_checksum="lora123",
        timestamp="2026-08-06T10:00:00",
        device="cuda:0",
        capability_metrics=compute_capability_metrics(records, _density()),
        # R6-EVAL-2 (§11.5.3): the five-piece outcome contract rides the
        # artifact; the schema-valid shape for a no-evidence bank is
        # NOT_INFORMATIVE (null value + frozen reason), never a 0.
        capability_outcomes=_outcomes_for(records),
        history_indices=[5, 6, 7],
        use_obstacle_projection=False,  # §7.4
        bank_id="bank123",  # bank provenance (v3, P1-3)
        offline_density=_density(),  # density provenance (P1-3)
    )


def test_artifact_schema_all_fields():
    a = _artifact()
    d = json.loads(json.dumps(a.to_dict()))
    for key in (
        "episodes", "checkpoint_hash", "config_hash", "contract_hash",
        "manifest_hash", "code_hash", "reward_version", "exec_steps",
        "observation_invalid_count", "missing_weight_keys",
        "unexpected_weight_keys", "lora_checksum", "timestamp", "device",
        "capability_metrics", "history_indices", "use_obstacle_projection",
        "bank_id", "offline_density_sha256",  # bank/density provenance (v3)
    ):
        assert key in d, f"missing §4.17 artifact field {key}"


def test_artifact_episode_level_not_aggregate():
    """§4.17: episodes are a list of paired records, not just aggregates."""
    a = _artifact()
    assert len(a.episodes) == 3
    ep = a.episodes[0]
    assert ep.bc["success"] is True
    assert ep.rl["collision"] is True
    assert ep.seed == 42 and ep.world == "w1"


def test_artifact_trace_rides_each_record():
    """ the raw per-step trace for both sides travels with the
    record and survives the JSON round-trip."""
    a = _artifact()
    d = json.loads(json.dumps(a.to_dict()))
    ep = d["episodes"][0]
    assert ep["trace"]["bc"], "bc side of the raw trace must be present"
    assert ep["trace"]["rl"], "rl side of the raw trace must be present"
    assert ep["trace"]["bc"][0]["step"] == 0
    assert ep["trace"]["rl"][0]["pos"] == [0.0, 0.0, 1.0]


def _single_role_artifact(ep: EpisodePairedRecord) -> EvalArtifact:
    """One-role artifact: capability metrics must follow the episode set
    (a role absent from episodes gets n_episodes=0), or the validator
    fails on role alignment before reaching the trace check."""
    return dataclasses.replace(
        _artifact(), episodes=[ep],
        capability_metrics=_caps_for([ep]),
    )


def test_trace_missing_hard_fails():
    """ a record without the per-step trace is aggregate-only —
    forbidden."""
    ep = dataclasses.replace(_paired(), trace={})
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="trace"):
        validate_eval_artifact(a)


def test_trace_side_missing_hard_fails():
    ep = dataclasses.replace(_paired(), trace={"bc": _trace()["bc"]})
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="trace\\[rl\\]"):
        validate_eval_artifact(a)


def test_trace_step_missing_key_hard_fails():
    """ every step entry must carry the full step observation —
    missing clearance means the trace cannot be audited."""
    bad = _trace()
    del bad["bc"][1]["clearance"]
    ep = dataclasses.replace(_paired(), trace=bad)
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="clearance"):
        validate_eval_artifact(a)


def test_artifact_validation_passes():
    validate_eval_artifact(_artifact())


def test_zero_episodes_hard_fails():
    """§4.17: aggregate-only artifacts are forbidden — zero episodes
    hard-fail."""
    a = dataclasses.replace(_artifact(), episodes=[])
    with pytest.raises(EvalArtifactError):
        validate_eval_artifact(a)


def test_episode_record_missing_keys_hard_fails():
    """§4.17: every episode's bc/rl record must carry the full
    success/collision/timeout/return/steps contract."""
    ep = dataclasses.replace(_paired(), bc={"success": True})
    a = dataclasses.replace(_artifact(), episodes=[ep])
    with pytest.raises(EvalArtifactError):
        validate_eval_artifact(a)


def test_missing_lora_checksum_hard_fails():
    """[JUDGMENT] LoRA missing -> hard fail (no warn-and-continue)."""
    a = dataclasses.replace(_artifact(), lora_checksum="")
    with pytest.raises(EvalArtifactError):
        validate_eval_artifact(a)


def test_missing_weight_keys_hard_fails():
    a = dataclasses.replace(_artifact(), missing_weight_keys=["actor_ft.0.weight"])
    with pytest.raises(EvalArtifactError):
        validate_eval_artifact(a)


def test_contract_mismatch_hard_fails():
    a = dataclasses.replace(_artifact(), contract_hash="MISMATCH")
    with pytest.raises(EvalArtifactError):
        validate_eval_artifact(a)


def test_observation_invalid_hard_fails():
    a = dataclasses.replace(_artifact(), observation_invalid_count=3)
    with pytest.raises(EvalArtifactError):
        validate_eval_artifact(a)


def test_capability_metrics_three_categories():
    a = _artifact()
    caps = a.capability_metrics
    assert set(caps) == {
        "anticipatory_avoidance",
        "fullspace_support_expansion",
        "dense_obstacle_ceiling",
    }


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda c: {**c, "anticipatory_avoidance": {
                **c["anticipatory_avoidance"], "sr_bc": True}},
            "must be a number",
        ),
        (
            lambda c: {**c, "anticipatory_avoidance": {
                **c["anticipatory_avoidance"], "sr_bc": float("nan")}},
            "must be finite",
        ),
        (
            lambda c: {**c, "anticipatory_avoidance": {
                **c["anticipatory_avoidance"], "sr_bc": float("inf")}},
            "must be finite",
        ),
        (
            lambda c: {**c, "anticipatory_avoidance": {
                k: v for k, v in c["anticipatory_avoidance"].items()
                if k != "cr_rl"}},
            "missing keys",
        ),
        (
            lambda c: {**c, "anticipatory_avoidance": {
                **c["anticipatory_avoidance"], "n_episodes": 7}},
            "role alignment",
        ),
        (
            lambda c: {**c, "anticipatory_avoidance": {
                **c["anticipatory_avoidance"], "sr_bc": 1.5}},
            r"in \[0, 1\]",
        ),
        (
            lambda c: {k: v for k, v in c.items()
                       if k != "dense_obstacle_ceiling"},
            "must be exactly",
        ),
    ],
)
def test_capability_validator_negative(mutate, match):
    """The artifact validator must hard-fail on malformed capability
    metrics — a poisoned metric must never validate into a PPO==BC
    result (bool/NaN/Inf, missing key, role misalignment, out-of-range
    sr/cr, wrong group set)."""
    caps = mutate(_caps_for(_three_role_records()))
    a = dataclasses.replace(_artifact(), capability_metrics=caps)
    with pytest.raises(EvalArtifactError, match=match):
        validate_eval_artifact(a)


def test_exec_steps_must_match_contract():
    a = dataclasses.replace(_artifact(), exec_steps=8)  # != contract 4
    with pytest.raises(EvalArtifactError):
        validate_eval_artifact(a, expected_exec_steps=4)


def test_recorded_noise_seed_must_rederive_from_provenance():
    """R5.8-E (N4) negative: a recorded policy_noise_seed that does NOT
    re-derive from the artifact's own (bank_id, episode_config_hash,
    replica) provenance is a provenance break — hard fail."""
    import dataclasses

    bad = dataclasses.replace(
        _artifact(), episodes=[
            dataclasses.replace(ep, policy_noise_seed=0)
            for ep in _artifact().episodes
        ]
    )
    with pytest.raises(EvalArtifactError, match="re-derive"):
        validate_eval_artifact(bad)


# ═══════════════════════════════════════════════════════════════════
# R6-E §11.6.4 (F-10): terminal_category 与 raw event flags 契约
# ═══════════════════════════════════════════════════════════════════

def test_record_missing_terminal_category_hard_fails():
    """§11.6.4: every record carries the mutually exclusive
    terminal_category — a record that cannot say HOW an episode ended is
    rejected (aggregate-only is forbidden)."""
    ep = dataclasses.replace(
        _paired(),
        rl={k: v for k, v in _paired().rl.items()
            if k != "terminal_category"},
    )
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="terminal_category"):
        validate_eval_artifact(a)


def test_record_missing_raw_flags_hard_fails():
    """The raw Gymnasium terminated/truncated flags must ride the record
    — the terminal_category alone is not enough (raw events are the
    §11.6.4 contract, and the category is re-derived from them)."""
    ep = dataclasses.replace(
        _paired(),
        bc={k: v for k, v in _paired().bc.items() if k != "terminated"},
    )
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="terminated"):
        validate_eval_artifact(a)


def test_terminal_category_not_in_enum_hard_fails():
    ep = dataclasses.replace(
        _paired(), rl={**_paired().rl, "terminal_category": "bogus"}
    )
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="terminal_category"):
        validate_eval_artifact(a)


def test_raw_flag_type_hard_fails():
    """bool-int (True/1) must not validate — type strictness is the
    contract (an int True would be re-derived as a bool elsewhere)."""
    ep = dataclasses.replace(
        _paired(), bc={**_paired().bc, "terminated": 1}
    )
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="terminated"):
        validate_eval_artifact(a)


def test_terminal_category_mismatch_re_derived_hard_fails():
    """§11.6.4: the stored category must equal the category re-derived
    from the raw flags — a record claiming "reached" on collision flags
    is an unmappable state and hard-fails (no priority-silent
    reclassification)."""
    ep = dataclasses.replace(
        _paired(), rl={**_paired().rl, "terminal_category": "reached"}
    )  # raw flags say collision
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="re-deriv|terminal"):
        validate_eval_artifact(a)


def test_terminal_category_cofire_cases_validate():
    """§11.6.4 acceptance: reached+truncated and collision+truncated
    co-fires map to a UNIQUE category (the terminated reason wins, never
    a timeout), and reached+collision maps to reached_collision — all
    three validate cleanly through the artifact."""
    from diffusion_policy_3d.env.episode_outcome import terminal_category

    cases = [
        # reached + truncated co-fire -> "reached"
        dict(terminated=True, truncated=True, success=True,
             collision=False, done_reason="reached",
             out_of_bounds=False, altitude_exceed=False),
        # collision + truncated co-fire -> "collision"
        dict(terminated=True, truncated=True, success=False,
             collision=True, done_reason="collision",
             out_of_bounds=False, altitude_exceed=False),
        # reached + collision co-fire -> "reached_collision"
        dict(terminated=True, truncated=False, success=True,
             collision=True, done_reason="collision",
             out_of_bounds=False, altitude_exceed=False),
        # pure timeout -> "timeout"
        dict(terminated=False, truncated=True, success=False,
             collision=False, done_reason="timeout",
             out_of_bounds=False, altitude_exceed=False),
    ]
    assert [terminal_category(**c) for c in cases] == [
        "reached", "collision", "reached_collision", "timeout",
    ]
    # and the mapped categories validate through the artifact
    for case in cases:
        ep = dataclasses.replace(
            _paired(),
            bc={**_paired().bc,
                "terminated": case["terminated"],
                "truncated": case["truncated"],
                "success": case["success"],
                "collision": case["collision"],
                "done_reason": case["done_reason"],
                "timeout": case["truncated"] and case["done_reason"] == "timeout",
                "terminal_category": terminal_category(**case)},
        )
        a = _single_role_artifact(ep)
        validate_eval_artifact(a)


def test_record_timeout_flag_must_match_done_reason():
    """§11.6.4: timeout 仅在 done_reason=="timeout" 时计 — 一个
    timeout=True 但 done_reason="reached" 的记录是旧语义残留
    （bool(truncated or loop_timeout)），硬失败。"""
    ep = dataclasses.replace(
        _paired(), bc={**_paired().bc, "timeout": True})
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="timeout"):
        validate_eval_artifact(a)


def test_record_rollout_cut_category_validates():
    """§11.6.4: rollout_cut 非 env termination — raw 标志全 False、
    done_reason="none" 的干净非 env 状态携带 terminal_category=
    "rollout_cut" 校验通过（re-derivation 用 other=stored 重放）。"""
    ep = dataclasses.replace(
        _paired(),
        bc={**_paired().bc, "terminated": False, "truncated": False,
            "success": False, "done_reason": "none",
            "terminal_category": "rollout_cut", "timeout": False},
    )
    a = _single_role_artifact(ep)
    validate_eval_artifact(a)


def test_record_rollout_cut_with_env_termination_hard_fails():
    """rollout_cut 与 env termination（terminated=True，raw flags 仍说
    reached）同现 → 类别与 raw 事件矛盾，硬失败。"""
    ep = dataclasses.replace(
        _paired(), bc={**_paired().bc, "terminal_category": "rollout_cut"})
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="rollout_cut|terminal"):
        validate_eval_artifact(a)


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-EVAL-2 (§11.5.3): five-piece outcome contract in the
# artifact schema + build_runtime_eval_artifact 生产接线
# ═══════════════════════════════════════════════════════════════════

def test_artifact_capability_outcomes_five_piece_schema():
    """§11.5.3: the artifact carries the five-piece outcome contract for
    all three groups — the outcome schema, not just the three-group
    numerics — and survives the JSON round-trip."""
    a = _artifact()
    assert set(a.capability_outcomes) == set(CAPABILITY_METRIC_KEYS)
    for group, pieces in a.capability_outcomes.items():
        assert set(pieces) == set(FIVE_PIECE_KEYS), group
        assert pieces["opportunity_count"] == 0
        assert pieces["value"] is None  # no-evidence null, never a 0
    d = json.loads(json.dumps(a.to_dict()))
    assert "capability_outcomes" in d
    assert d["capability_outcomes"]["anticipatory_avoidance"][
        "opportunity_count"] == 0


def test_artifact_outcomes_missing_group_hard_fails():
    """R6-EVAL-2: 五件套缺一组是 schema 违反 —— 与三组 numerics 同级别
    硬失败，绝不允许静默缺省。"""
    partial = {k: v for k, v in _outcomes_for([]).items()
               if k != "dense_obstacle_ceiling"}
    a = dataclasses.replace(_artifact(), capability_outcomes=partial)
    with pytest.raises(EvalArtifactError, match="capability_outcomes"):
        validate_eval_artifact(a)


def test_artifact_outcomes_invalid_counts_hard_fail():
    """五件套 count 必须是非负 int（拒 bool）—— 负 opportunity 或
    float count 都是捏造语义。"""
    pieces = _outcomes_for([])["anticipatory_avoidance"]
    bad = dict(pieces, opportunity_count=-1)
    a = dataclasses.replace(
        _artifact(),
        capability_outcomes={
            "anticipatory_avoidance": bad,
            "fullspace_support_expansion": dict(pieces),
            "dense_obstacle_ceiling": dict(pieces),
        },
    )
    with pytest.raises(EvalArtifactError, match="opportunity_count"):
        validate_eval_artifact(a)
    bad_float = dict(pieces, exposure_count=1.5)
    a2 = dataclasses.replace(
        _artifact(),
        capability_outcomes={
            "anticipatory_avoidance": bad_float,
            "fullspace_support_expansion": dict(pieces),
            "dense_obstacle_ceiling": dict(pieces),
        },
    )
    with pytest.raises(EvalArtifactError, match="exposure_count"):
        validate_eval_artifact(a2)


def test_artifact_outcomes_events_require_value_hard_fail():
    """event_count>0 必须带有限 value（否则 value 语义模糊）；
    event_count==0 必须 value null + validity_reason（诚实空态，§11.5.5
    "null, never a 0 score"）。"""
    pieces = _outcomes_for([])["anticipatory_avoidance"]
    with_value = dict(pieces, opportunity_count=1, exposure_count=1,
                      event_count=1, value=None, validity_reason=None)
    a = dataclasses.replace(
        _artifact(),
        capability_outcomes={
            "anticipatory_avoidance": with_value,
            "fullspace_support_expansion": dict(pieces),
            "dense_obstacle_ceiling": dict(pieces),
        },
    )
    with pytest.raises(EvalArtifactError, match="value"):
        validate_eval_artifact(a)
    no_reason = dict(pieces, validity_reason=None)
    a2 = dataclasses.replace(
        _artifact(),
        capability_outcomes={
            "anticipatory_avoidance": no_reason,
            "fullspace_support_expansion": dict(pieces),
            "dense_obstacle_ceiling": dict(pieces),
        },
    )
    with pytest.raises(EvalArtifactError, match="validity_reason"):
        validate_eval_artifact(a2)


def test_artifact_outcomes_nonfinite_value_hard_fail():
    """P1-4: NaN/Inf 永远是非法指标 —— 五件套 value 同规则。"""
    pieces = _outcomes_for([])["anticipatory_avoidance"]
    nan_value = dict(pieces, opportunity_count=1, exposure_count=1,
                     event_count=1, value=float("nan"),
                     validity_reason=None)
    a = dataclasses.replace(
        _artifact(),
        capability_outcomes={
            "anticipatory_avoidance": nan_value,
            "fullspace_support_expansion": dict(pieces),
            "dense_obstacle_ceiling": dict(pieces),
        },
    )
    with pytest.raises(EvalArtifactError, match="finite"):
        validate_eval_artifact(a)


def _runtime_builder_inputs(ep_hashes=None):
    """build_runtime_eval_artifact 的最小生产形状输入（records 三角色、
    合法 trace、真实 provenance）——与 _artifact() 同一批记录。

    ep_hashes: 可选的 {role: hash} 映射 —— wiring 测试需要三角色
    不同 episode_config_hash（默认共享 e0*32，恰好镜像 v3 bank
    三机会同 hash 的退化情形）。
    """
    from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract

    roles = ["anticipatory", "support_expansion", "dense_recovery"]
    records = _three_role_records()
    if ep_hashes:
        records = [_paired(r, ep_hash=ep_hashes[r]) for r in roles]

    return dict(
        records=records,
        checkpoint_hash="ckpt123",
        config_dict={"config_layer": {}},
        contract=RLSystemContract(),
        runtime={
            "manifest_hash": "manifestabc",
            "source_file_hashes": {},
            "reward_version": "v16",
            "timestamp": "2026-08-19T10:00:00",
            "device": "cpu",
            "identity_mode": "strict",
            "source_file_delta": None,
            "project_git_commit": "abc123",
        },
        lora_checksum="lora123",
        observation_invalid_count=0,
        missing_weight_keys=[],
        unexpected_weight_keys=[],
        history_indices=[5, 6, 7],
        offline_density=_density(),
        bank_id="bank123",
    )


def test_runtime_builder_wires_opportunity_map_to_outcomes():
    """Gate B R6-EVAL-2: build_runtime_eval_artifact 必须从
    opportunity_by_entry 计算五件套 —— 生产 artifact 的 capability
    gate 携带真实 opportunity 证据（三组全部 INFORMATIVE）。"""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        build_runtime_eval_artifact,
    )

    inputs = _runtime_builder_inputs(ep_hashes={
        "anticipatory": "e1" * 32,
        "support_expansion": "e2" * 32,
        "dense_recovery": "e3" * 32,
    })
    opp_map = {}
    for r in inputs["records"]:
        if r.role == "anticipatory":
            opp = CapabilityOpportunity(
                valid=True, risk_shell_crossing=True,
                anticipatory_provenance="nominal_path_geometry")
        elif r.role == "support_expansion":
            opp = CapabilityOpportunity(
                valid=True, support_stratum="outside",
                support_provenance="offline_density")
        else:
            opp = CapabilityOpportunity(
                valid=True, topology_class="trap",
                trap_opportunity_count=1, dense_provenance="world_topology")
        opp_map[r.episode_config["episode_config_hash"]] = opp
    artifact = build_runtime_eval_artifact(
        **inputs, opportunity_by_entry=opp_map)
    assert set(artifact.capability_outcomes) == set(CAPABILITY_METRIC_KEYS)
    statuses = capability_informativeness(artifact.capability_outcomes)
    for group in CAPABILITY_METRIC_KEYS:
        assert statuses[group]["status"] == "INFORMATIVE", group
        assert artifact.capability_outcomes[group]["opportunity_count"] == 1


def test_runtime_builder_without_opportunity_map_is_not_informative():
    """Gate B R6-EVAL-2: 缺 opportunity map（v3 bank 无 opportunity 块）
    → 五件套全部 NOT_INFORMATIVE，绝无捏造 value —— 忘传 map 只会让
    gate 更保守（BLOCKED），不可能虚报能力分数。"""
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        build_runtime_eval_artifact,
    )

    artifact = build_runtime_eval_artifact(**_runtime_builder_inputs())
    statuses = capability_informativeness(artifact.capability_outcomes)
    for group in CAPABILITY_METRIC_KEYS:
        assert statuses[group]["status"] == "NOT_INFORMATIVE", group
        pieces = artifact.capability_outcomes[group]
        assert pieces["opportunity_count"] == 0
        assert pieces["value"] is None
