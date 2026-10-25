"""Role-first episode ticket sampling and exact resume identity."""

from collections import Counter
import copy

import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_sampler import (
    RoleBalancedTaskSampler,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import (
    EpisodeTaskSpec,
    TRAINABLE_ROLES,
)


def _tasks():
    return tuple(
        EpisodeTaskSpec(
            "task_" + role,
            "train_" + role,
            role + ".json",
            role,
            index + 1,
            True,
            identity_hash=(str(index + 1) * 64)[:64],
        )
        for index, role in enumerate(TRAINABLE_ROLES)
    )


def test_every_role_prefix_is_balanced():
    sampler = RoleBalancedTaskSampler(_tasks(), seed=17)
    counts = Counter()
    for _ in range(1024):
        counts[sampler.sample().role] += 1
        values = [counts.get(role, 0) for role in TRAINABLE_ROLES]
        assert max(values) - min(values) <= 1
    assert counts == Counter({role: 256 for role in TRAINABLE_ROLES})


def test_sampler_cold_resume_is_exact_and_identity_checked():
    sampler = RoleBalancedTaskSampler(_tasks(), seed=19)
    for _ in range(13):
        sampler.sample()
    state = sampler.state_dict()
    expected = [sampler.sample().task_id for _ in range(20)]
    restored = RoleBalancedTaskSampler(_tasks(), seed=999)
    restored.load_state_dict(state)
    assert [restored.sample().task_id for _ in range(20)] == expected
    bad = copy.deepcopy(state)
    bad["task_hashes_by_role"][TRAINABLE_ROLES[0]][0] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        restored.load_state_dict(bad)
