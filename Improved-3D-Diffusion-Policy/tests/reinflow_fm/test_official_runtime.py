from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch import nn

from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.lora import install_lora
from diffusion_policy_3d.rl.reinflow_fm.condition_adapter import FMConditionBatch
from diffusion_policy_3d.rl.reinflow_fm.policy_bridge import (
    IDP3FlowPolicyBridge,
    condition_to_official_dict,
)
from diffusion_policy_3d.rl.reinflow_fm.runtime import (
    PPOFlowBuildConfig,
    build_idp3_ppo_flow,
)


class _NativePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(11, 11, bias=False))
        self.horizon = 16
        self.action_dim = 11
        self.sigma_min = 0.005
        self.fm_t_scale = 50
        self.cfg_lambda = 1.5

    def _compute_cfg_velocity(
        self, action, t_input, global_cond, global_cond_uncond, local_cond
    ):
        goal_signal = (global_cond - global_cond_uncond).mean(dim=1)
        return self.model(action) + goal_signal[:, None, None] * 0.01


class _Critic(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.value = nn.Linear(cond_dim, 1)

    def forward(self, cond):
        return self.value(cond["state"])


class OfficialRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[2]
        cls.vendor = repo / "third_party" / "reinflow_official_e722e15"
        sys.path.insert(0, str(cls.vendor))
        from model.flow.ft_ppo.ppoflow import PPOFlow
        from model.flow.mlp_flow import NoisyFlowMLP

        cls.PPOFlow = PPOFlow
        cls.NoisyFlowMLP = NoisyFlowMLP

    @classmethod
    def tearDownClass(cls):
        vendor_text = str(cls.vendor)
        sys.path[:] = [entry for entry in sys.path if entry != vendor_text]

    def test_official_chain_and_recomputed_logprob_have_unit_first_ratio(self):
        torch.manual_seed(31)
        native = _NativePolicy()
        lora = install_lora(
            native.model,
            rank=2,
            alpha=2.0,
            expected_linear=1,
            expected_conv1d=0,
        )
        bridge = IDP3FlowPolicyBridge(
            native, lora_bundle=lora, lora_enabled=False, cond_enc_dim=4
        )
        symbols = SimpleNamespace(PPOFlow=self.PPOFlow, NoisyFlowMLP=self.NoisyFlowMLP)
        runtime = build_idp3_ppo_flow(
            symbols,
            bridge,
            _Critic(cond_dim=12),
            PPOFlowBuildConfig(
                device="cpu",
                act_min=-20.0,
                act_max=20.0,
                obs_dim=12,
                cond_steps=3,
                noise_scheduler_type="const",
                inference_steps=4,
                ft_denoising_steps=4,
                randn_clip_value=10.0,
                min_sampling_denoising_std=0.1,
                min_logprob_denoising_std=0.1,
                max_logprob_denoising_std=0.1,
                logprob_min=-1.0e9,
                logprob_max=1.0e9,
                clip_ploss_coef=0.2,
                clip_ploss_coef_base=0.001,
                clip_ploss_coef_rate=3.0,
                clip_vloss_coef=0.0,
                denoised_clip_value=20.0,
                time_dim_explore=4,
                learn_explore_time_embedding=True,
                use_time_independent_noise=True,
                noise_hidden_dims=(8,),
                logprob_debug_sample=False,
                logprob_debug_recalculate=False,
                explore_net_activation_type="Tanh",
            ),
        )
        condition = FMConditionBatch(
            cond=torch.randn(3, 12),
            uncond=torch.randn(3, 12),
            noise_cond=torch.randn(3, 4),
            raw_obs_sha256=("a" * 64,) * 3,
            encoder_sha256="b" * 64,
            normalizer_sha256="c" * 64,
            n_obs_steps=3,
        )
        official_cond = condition_to_official_dict(condition)

        _, chain, old_logprob = runtime.get_actions(
            official_cond,
            eval_mode=False,
            save_chains=True,
            normalize_denoising_horizon=False,
            normalize_act_space_dimension=False,
            clip_intermediate_actions=False,
            account_for_initial_stochasticity=True,
        )
        new_logprob = runtime.get_logprobs(
            official_cond,
            chain,
            get_entropy=False,
            normalize_denoising_horizon=False,
            normalize_act_space_dimension=False,
            clip_intermediate_actions=False,
            verbose_entropy_stats=False,
            account_for_initial_stochasticity=True,
            get_chains_stds=False,
        )
        ratio = torch.exp(new_logprob - old_logprob)

        self.assertIsInstance(runtime, self.PPOFlow)
        self.assertIsInstance(runtime.actor_ft, self.NoisyFlowMLP)
        self.assertEqual(tuple(chain.shape), (3, 5, 16, 11))
        self.assertLess(float((ratio - 1.0).abs().max()), 1.0e-6)
        trainable_policy = [
            name
            for name, parameter in runtime.actor_ft.policy.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_policy)
        self.assertTrue(all(name.endswith(("lora_a", "lora_b")) for name in trainable_policy))


if __name__ == "__main__":
    unittest.main()
