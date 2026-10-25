import hashlib
from pathlib import Path
import tempfile
import unittest

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def nested(value):
    if isinstance(value, dict):
        return AttrDict({k: nested(v) for k, v in value.items()})
    return value


def valid_r1_config():
    return nested(dict(
        reward_version="v2", checkpoint="bc.ckpt", device="cpu", vendor_root="vendor",
        admission_manifest="manifest.json", worlds_train="worlds", output_dir="r1",
        reward=dict(H_e=4, max_episode_steps=60, gamma_pbrs=.99,
                    r_reach=100., r_failure=50., newbest_budget=10.,
                    risk_weight=.5, time_cost=.05, d_safe=.45,
                    collision_margin_xy=.1, ground_safe=1., ground_contact=.2),
        run=dict(phase="formal", seed=202609201, updates=256,
                 rollout_transitions=1024, schedule_label="S2",
                 checkpoint_updates=list(range(0, 257, 8)),
                 expected_optimizer_commits_per_outer_update=40,
                 checkpoint_every_updates=8),
        optimizer=dict(actor_lr=2e-5, actor_lr_end=1e-5, critic_lr=.001,
                       weight_decay=.01),
        dppo=dict(gamma_environment=.99, gae_lambda=.95,
                  gamma_denoising=.8, clip_ploss_coef=.1,
                  min_sampling_std=.1, min_logprob_std=.1,
                  value_loss_coefficient=.5, entropy_loss_coefficient=0.,
                  denoising_schedule="full_joint_accum_batch1280",
                  pair_passes=5, logical_batch_pairs=1280, microbatch_pairs=128),
        environment=dict(max_episode_steps=60, disturb_enable=True,
                         disturb_prob=.2, disturb_pos_scale=.3,
                         exec_noise_scale=0., mid_state_init_ratio=.3,
                         mid_state_pos_scale=.5, max_reset_attempts=50,
                         goal_tolerance_3d=.5, reject_all_no_ray_hit=False,
                         max_no_ray_hit_fraction=.999,
                         pointcloud_calibration_sha256="a" * 64),
        lora=dict(rank=8, alpha=8.), critic=dict(hidden_dim=256),
        gates=dict(max_approx_kl=.2, ratio_mean_min=.2,
                   ratio_mean_max=5., max_grad_norm=5.),
    ))


class R1ConfigTest(unittest.TestCase):
    def test_first_event_sources_change_r1_environment_provenance(self):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import build_environment_provenance

        base_files = (
            "diffusion_policy_3d/env/collision_geom.py",
            "diffusion_policy_3d/env/kinematic_uav_env.py",
            "diffusion_policy_3d/env/traj_utils.py",
            "diffusion_policy_3d/model/common/synth_pointcloud.py",
        )
        r1_files = (
            "diffusion_policy_3d/env/reward_v2_support/environment.py",
            "diffusion_policy_3d/env/reward_v2_support/first_event.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in base_files + r1_files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original\n")
            for relative in r1_files:
                with self.subTest(relative=relative):
                    before = build_environment_provenance(root, reward_version="v2")
                    (root / relative).write_bytes(b"modified\n")
                    after = build_environment_provenance(root, reward_version="v2")
                    self.assertEqual(before["source_sha256"][relative],
                                     hashlib.sha256(b"original\n").hexdigest())
                    self.assertEqual(after["source_sha256"][relative],
                                     hashlib.sha256(b"modified\n").hexdigest())
                    self.assertNotEqual(before["content_hash"], after["content_hash"])
                    (root / relative).write_bytes(b"original\n")

    def test_r1_formal_config_accepts_only_frozen_reward_coefficients(self):
        config = valid_r1_config()
        self.assertEqual(validate_v2_config(config).schedule, "full_joint_accum_batch1280")
        config.reward["time_cost"] = .1
        with self.assertRaises(ValueError):
            validate_v2_config(config)

    def test_r1_rejects_mismatched_environment_discount(self):
        config = valid_r1_config()
        config.dppo["gamma_environment"] = .98
        with self.assertRaisesRegex(ValueError, "gamma_environment"):
            validate_v2_config(config)


if __name__ == "__main__":
    unittest.main()
