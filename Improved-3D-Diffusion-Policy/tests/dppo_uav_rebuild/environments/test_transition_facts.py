import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import (
    EpisodeTaskSpec,
    TransitionFacts,
)


@pytest.mark.parametrize(
    "collision,early,reached,limit,reason",
    [
        (True, True, True, True, "collision"),
        (False, True, True, True, "early_death"),
        (False, False, True, True, "reached"),
        (False, False, False, True, "time_limit"),
        (False, False, False, False, "none"),
    ],
)
def test_terminal_precedence(collision, early, reached, limit, reason):
    facts = TransitionFacts.resolve(
        collided=collision,
        early_death=early,
        reached=reached,
        time_limit=limit,
    )
    assert facts.done_reason == reason


def test_eval_world_cannot_enter_training():
    with pytest.raises(ValueError, match="eval-only"):
        EpisodeTaskSpec("bad", "w4", "w4.json", "anticipatory", 1, True)

