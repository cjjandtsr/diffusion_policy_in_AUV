import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.evaluation.paired_bank import PairedEpisodeRecord


def _record(**updates):
    values = dict(
        task_id="t", world_id="w", role="anticipatory", seed=1, policy="bc",
        success=True, collision=False, episode_return=0.0,
        avoidance_onset_distance=0.0, support_cells=0, steps=1,
        goal_xyz=(0.0, 0.0, 0.0), final_position_xyz=(0.3, 0.0, 0.4),
        final_distance_3d=0.5, final_error_xy=0.3, final_error_z=0.4,
        terminal_safety_flags=(), observation_validity_summary=(("valid", True),),
        done_reason="reached",
    )
    values.update(updates)
    return PairedEpisodeRecord(**values)


def test_safe_arrival_is_three_dimensional_distance_and_safety_only():
    assert _record().success is True
    with pytest.raises(ValueError, match="arrival and collision"):
        _record(collision=True, success=False, terminal_safety_flags=())
    with pytest.raises(ValueError, match="success must mean"):
        _record(success=True, terminal_safety_flags=(("invalid_state", True),))


def test_wrong_reported_distance_is_rejected():
    with pytest.raises(ValueError, match="endpoint geometry"):
        _record(final_distance_3d=0.49)
