from omegaconf import OmegaConf
import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config


def _config(label="S0", seed=202609101):
    schedule = "sampled_k" if label == "S0" else "full_joint_accum_batch1280"
    logical = 128 if label == "S0" else 1280
    return OmegaConf.create({
        "reward_version": "v17", "reward": {"H_e": 4},
        "optimizer": {"actor_lr": 2e-5, "actor_lr_end": 1e-5},
        "run": {"phase": "pilot", "seed": seed, "updates": 32,
                "rollout_transitions": 1024, "schedule_label": label,
                "checkpoint_updates": [0, 8, 16, 24, 32],
                "expected_optimizer_commits_per_outer_update": 40},
        "dppo": {"denoising_schedule": schedule, "pair_passes": 5, "clip_ploss_coef": 0.1,
                 "logical_batch_pairs": logical, "microbatch_pairs": 128},
        "environment": {
            "max_episode_steps": 60, "disturb_enable": True,
            "disturb_prob": 0.2, "disturb_pos_scale": 0.3,
            "exec_noise_scale": 0.0, "mid_state_init_ratio": 0.3,
            "mid_state_pos_scale": 0.5, "max_reset_attempts": 50,
            "reject_all_no_ray_hit": False,
            "max_no_ray_hit_fraction": 0.9,
        },
    })


def test_s0_and_s2_contracts_are_exact():
    assert validate_v2_config(_config("S0")).expected_optimizer_commits_per_outer_update == 40
    assert validate_v2_config(_config("S2")).logical_batch_pairs == 1280


def test_legacy_minibatch_and_wrong_commit_count_are_rejected():
    config = _config()
    config.run.minibatch_size = 128
    with pytest.raises(ValueError, match="forbidden"):
        validate_v2_config(config)


def test_shipped_pilot_template_is_fail_closed_until_bc_calibration():
    config = OmegaConf.load(
        "diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_v17_pilot_s0_seed1_v2.yaml"
    )
    with pytest.raises(ValueError, match="requires frozen BC calibration"):
        validate_v2_config(config)
    config.environment.max_no_ray_hit_fraction = 0.9
    config.environment.pointcloud_calibration_sha256 = "a" * 64
    assert validate_v2_config(config).schedule == "sampled_k"
    config = _config()
    config.run.expected_optimizer_commits_per_outer_update = 39
    with pytest.raises(ValueError, match="mismatch"):
        validate_v2_config(config)


@pytest.mark.parametrize("label", ["s0", "s2"])
def test_formal_template_declares_every_eight_update_checkpoint(label):
    config = OmegaConf.load(
        "diffusion_policy_3d/rl/dppo_uav_rebuild/config/"
        "train_e5_dppo_v17_formal_{}_template_v2.yaml".format(label)
    )
    config.environment.max_no_ray_hit_fraction = 0.9
    config.environment.pointcloud_calibration_sha256 = "a" * 64
    contract = validate_v2_config(config)
    assert tuple(config.run.checkpoint_updates) == tuple(range(0, 257, 8))
    assert contract.expected_optimizer_commits_per_outer_update == 40


def test_formal_budget_is_configurable_without_changing_r1_reward_or_schedule():
    config = OmegaConf.load(
        "diffusion_policy_3d/rl/dppo_uav_rebuild/config/"
        "train_e5_dppo_r1_s2_seed202609203_resetfix_v1.yaml")
    config.run.updates = 384
    config.run.checkpoint_updates = list(range(0, 385, 8))
    config.result_dir = "/home/cjj/AUV/program/results/routea_training/test_384"
    assert validate_v2_config(config).logical_batch_pairs == 1280
    config.reward.time_cost = .1
    with pytest.raises(ValueError, match="reward coefficients"):
        validate_v2_config(config)


@pytest.mark.parametrize("updates", [248, 383])
def test_formal_budget_keeps_minimum_and_eight_update_cadence(updates):
    config = OmegaConf.load(
        "diffusion_policy_3d/rl/dppo_uav_rebuild/config/"
        "train_e5_dppo_r1_s2_seed202609203_resetfix_v1.yaml")
    config.run.updates = updates
    config.run.checkpoint_updates = list(range(0, updates + 1, 8))
    with pytest.raises(ValueError, match="formal requires"):
        validate_v2_config(config)


def test_fresh_384_config_changes_only_budget_and_output_locations():
    folder = "diffusion_policy_3d/rl/dppo_uav_rebuild/config/"
    original = OmegaConf.load(folder + "train_e5_dppo_r1_s2_seed202609203_resetfix_v1.yaml")
    fresh = OmegaConf.load(folder + "train_e5_dppo_r1_s2_seed202609203_fresh384_v1.yaml")
    assert validate_v2_config(fresh).expected_optimizer_commits_per_outer_update == 40
    before = OmegaConf.to_container(original, resolve=True)
    after = OmegaConf.to_container(fresh, resolve=True)
    assert after.pop("result_dir").startswith("/home/cjj/AUV/program/")
    assert after.pop("output_dir") != before.pop("output_dir")
    assert after["run"].pop("updates") == 384
    assert before["run"].pop("updates") == 256
    assert after["run"].pop("checkpoint_updates") == list(range(0, 385, 8))
    assert before["run"].pop("checkpoint_updates") == list(range(0, 257, 8))
    assert before == after
