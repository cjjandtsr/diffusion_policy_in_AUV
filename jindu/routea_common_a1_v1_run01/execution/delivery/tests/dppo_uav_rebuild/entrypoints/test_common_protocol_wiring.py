"""New protocol must be selected coherently, with legacy configs unchanged."""

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import pipeline
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.select import select_reward
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import RouteAEnvironmentFactory


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609203_resetfix_v1.yaml"


def common_config(logical=2560):
    config = OmegaConf.load(BASE)
    config.task_protocol = {"name": "open_space_finite_v1", "space_cost_budget": 1.}
    config.run.schedule_label = "S2_common"
    config.dppo.denoising_schedule = "full_joint_accum"
    config.dppo.logical_batch_pairs = logical
    config.run.expected_optimizer_commits_per_outer_update = 51200 // logical
    config.run.checkpoint_updates = [0, 1] + list(range(8, 257, 8))
    return config


@pytest.mark.parametrize("logical,commits", [(1280, 40), (2560, 20)])
def test_two_common_schedules_validate_and_select_five_reward_terms(logical, commits):
    config = common_config(logical)
    contract = validate_v2_config(config)
    assert contract.expected_optimizer_commits_per_outer_update == commits
    assert select_reward(config, ROOT).term_names == ("terminal", "newbest", "risk", "time", "space")
    assert "space" not in select_reward(OmegaConf.load(BASE), ROOT).term_names


@pytest.mark.parametrize("change", ["legacy_schedule", "wrong_protocol", "negative_cost", "wrong_horizon"])
def test_protocol_rejects_mixed_or_invalid_contract(change):
    config = common_config()
    if change == "legacy_schedule":
        config.run.schedule_label = "S2"
    elif change == "wrong_protocol":
        config.task_protocol.name = "typo"
    elif change == "negative_cost":
        config.task_protocol.space_cost_budget = -1.
    else:
        config.environment.max_episode_steps = 61
    with pytest.raises((ValueError, TypeError)):
        validate_v2_config(config)


def test_factory_passes_open_space_and_cost_into_actual_environment():
    from types import SimpleNamespace
    task = SimpleNamespace(task_id="one", world_id="fixture", obstacle_json="fixture.json", role="dense_recovery", seed=1)
    reward = select_reward(common_config(), ROOT)
    with patch.object(RouteAEnvironmentFactory, "_build_tasks", return_value=(task,)):
        factory = RouteAEnvironmentFactory(manifest_path="fixture", worlds_train_dir="fixture", reward=reward,
                                           device="cpu", base_seed=1, terminate_at_space_limits=False,
                                           space_cost_budget=1.)
    with patch("diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory.KinematicUAVEnv") as constructor:
        factory.make(task)
    assert constructor.call_args.kwargs["terminate_at_space_limits"] is False
    assert constructor.call_args.kwargs["space_cost_budget"] == 1.


def test_finite_deadline_overrides_collection_cut_but_legacy_still_bootstraps():
    assert hasattr(pipeline, "_bootstrap_allowed")
    assert pipeline._bootstrap_allowed(False, False, finite_task=True)
    assert not pipeline._bootstrap_allowed(False, True, finite_task=True)
    assert not pipeline._bootstrap_allowed(True, False, finite_task=True)
    assert pipeline._bootstrap_allowed(False, True, finite_task=False)


def test_delivered_configs_differ_only_in_batch_commits_and_new_output_paths():
    configs = []
    for logical in (1280, 2560):
        c = OmegaConf.load(BASE.parent / ("train_e5_dppo_common_b%d_seed202609203_v1.yaml" % logical))
        validate_v2_config(c)
        assert c.output_dir.endswith("routea_common_b%d_seed202609203_v1" % logical)
        assert c.result_dir.startswith("/home/cjj/AUV/program/")
        assert c.task_protocol.space_cost_budget == 1.
        c.pop("output_dir")
        c.pop("result_dir")
        c.run.pop("expected_optimizer_commits_per_outer_update")
        c.dppo.pop("logical_batch_pairs")
        configs.append(OmegaConf.to_container(c))
    assert configs[0] == configs[1]


def test_first_real_update_can_checkpoint_and_resume_without_changing_budget():
    c = common_config()
    assert pipeline._should_save_checkpoint(c, 1)
    assert pipeline._execution_last_update(c, 1, 1) == 1
    assert c.run.updates == 256 and pipeline._actor_lr_for_update(c, 1) == 2e-5
