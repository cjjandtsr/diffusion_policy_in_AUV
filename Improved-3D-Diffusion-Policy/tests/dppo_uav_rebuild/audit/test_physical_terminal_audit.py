"""Catch lost physical failure causes without changing canonical PPO terminals."""

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.training_audit import TrainingAudit
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import TransitionFacts


def _record(audit, canonical, raw, *, partial=False, role="dense_recovery"):
    facts = TransitionFacts.resolve(
        collided=canonical == "collision",
        early_death=canonical == "early_death",
        reached=canonical == "reached",
        time_limit=canonical == "time_limit",
        raw_done_reason=raw,
    )
    audit.record_transition(
        task_id="fixture", task_role=role, reward=-1.0, value=0.0,
        next_value=0.0, terminated=facts.terminated, truncated=facts.truncated,
        collection_boundary=partial,
        info={
            "done_reason": facts.done_reason,
            "transition_facts": facts,
            "reward_terms": {
                "base": -1.0, "pbrs": 0.0, "onset": 0.0,
                "future_clearance": 0.0, "trap": 0.0, "flip": 0.0,
                "support": 0.0, "guard": 0.0,
            },
        },
    )


def _finish(audit, count):
    return audit.finalize({
        "rewards": -torch.ones(count), "values": torch.zeros(count),
        "next_values": torch.zeros(count), "advantages": -torch.ones(count),
        "returns": -torch.ones(count),
    })


def test_physical_causes_separate_ground_obstacles_and_flight_envelope():
    audit = TrainingAudit()
    for canonical, raw in [
        ("collision", "crash_ground"),
        ("collision", "collide(box_a,clr=0.00<0.1)"),
        ("collision", "collide(cylinder_b,segment_clr=0.02<0.1)"),
        ("collision", "penetrate_flyover(box_c)"),
        ("early_death", "out_of_bounds"),
        ("early_death", "altitude_exceed"),
        ("reached", "reached"),
        ("time_limit", "none"),
    ]:
        _record(audit, canonical, raw)
    _record(audit, "none", "none", partial=True, role="support_expansion")
    payload = _finish(audit, 9)
    episodes = payload["episodes"]
    assert episodes["completed"] == 8
    assert episodes["partial_segments"] == 1
    assert episodes["physical_done_reason_counts"] == {
        "crash_ground": 1, "obstacle_collision": 3, "out_of_bounds": 1,
        "altitude_exceed": 1, "reached": 1, "time_limit": 1,
    }
    assert episodes["physical_done_reason_rates"]["obstacle_collision"] == pytest.approx(3 / 8)
    assert episodes["physical_done_reason_counts_by_role"] == {
        "dense_recovery": episodes["physical_done_reason_counts"],
    }
    assert episodes["done_reason_counts"] == {
        "collision": 4, "early_death": 2, "reached": 1,
        "time_limit": 1, "collection_boundary": 1,
    }
    assert payload["reward"]["total"]["sum"] == -9.0
    assert payload["learning_targets"]["returns"]["sum"] == -9.0


@pytest.mark.parametrize("canonical,raw,want", [
    ("collision", None, "collision_unspecified"),
    ("early_death", None, "early_death_unspecified"),
    ("collision", "out_of_bounds", "collision_unspecified"),
    ("early_death", "crash_ground", "early_death_unspecified"),
    ("reached", "none", "reached"),
    ("time_limit", "none", "time_limit"),
])
def test_missing_or_conflicting_raw_reason_is_not_invented(canonical, raw, want):
    audit = TrainingAudit()
    _record(audit, canonical, raw)
    episodes = _finish(audit, 1)["episodes"]
    assert episodes["physical_done_reason_counts"] == {want: 1}
    assert episodes["done_reason_counts"] == {canonical: 1}


def test_collection_cutoff_is_not_counted_as_a_physical_failure():
    audit = TrainingAudit()
    _record(audit, "none", "none", partial=True)
    episodes = _finish(audit, 1)["episodes"]
    assert episodes["physical_done_reason_counts"] == {}
    assert episodes["physical_done_reason_rates"] == {}
    assert episodes["physical_done_reason_counts_by_role"] == {}
