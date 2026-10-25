"""The common evaluator reads the saved budget; A1 is not legacy S2."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from scripts.routea_paired_support.runtime import load_actor
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _canonical_hash
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.save import save_checkpoint
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import CheckpointHeader
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import restore_checkpoint
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.value import TemporalValueNetwork


ROOT = Path(__file__).resolve().parents[2]


def _config(logical):
    c = OmegaConf.load(ROOT / "diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609203_resetfix_v1.yaml")
    c.task_protocol = {"name": "open_space_finite_v1", "space_cost_budget": 1.}
    c.run.schedule_label = "S2_common"
    c.dppo.denoising_schedule = "full_joint_accum"
    c.dppo.logical_batch_pairs = logical
    c.run.expected_optimizer_commits_per_outer_update = 51200 // logical
    c.run.checkpoint_updates = [0, 1] + list(range(8, 257, 8))
    return OmegaConf.to_container(c, resolve=True)


class Bundle:
    identity = SimpleNamespace(content_hash="b" * 64)

    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.loaded = False

    def named_adapter_parameters(self):
        yield "actor", self.parameter

    def adapter_state_dict(self):
        return {"actor": self.parameter.detach().clone()}

    def load_adapter_state_dict(self, state):
        self.loaded = True


@pytest.mark.parametrize("logical", [1280, 2560])
def test_save_and_common_evaluator_load_actual_budget(tmp_path, logical):
    config = _config(logical)
    bundle = Bundle()
    critic = TemporalValueNetwork(hidden_dim=32, episode_context=True)
    with torch.no_grad():
        critic.context_weight.fill_(.125)
    expected_critic = {k: v.clone() for k, v in critic.state_dict().items()}
    actor_optimizer = torch.optim.Adam([bundle.parameter])
    critic_optimizer = torch.optim.Adam(critic.parameters())
    counters = SimpleNamespace(env_steps=32768, committed_updates=32 * (51200 // logical))
    header = CheckpointHeader(2, {"config_hash": _canonical_hash(config)}, "a" * 64, "b" * 64, 32)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, header=header, lora_bundle=bundle, critic=critic,
                    actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                    counters=counters,
                    training_config=config)
    state = torch.load(checkpoint / "state.pt", weights_only=True)
    assert state["training_config"] == config
    descriptor = {"checkpoint_path": str(checkpoint), "update": 32}
    with patch("scripts.routea_paired_support.runtime.verify_checkpoint", return_value={
        "lora_identity_hash": "b" * 64, "runtime_identity": header.runtime_identity}):
        load_actor(ROOT, SimpleNamespace(lora_bundle=bundle), descriptor)
    assert bundle.loaded
    expected_random = torch.rand(4)
    for parameter in critic.parameters():
        parameter.data.zero_()
    counters.env_steps = 0
    restore_checkpoint(checkpoint, runtime_identity=SimpleNamespace(content_hash="a" * 64),
                       lora_identity=bundle.identity, lora_bundle=bundle, critic=critic,
                       actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                       counters=counters)
    assert counters.env_steps == 32768
    assert all(torch.equal(critic.state_dict()[k], v) for k, v in expected_critic.items())
    assert torch.equal(torch.rand(4), expected_random)


def test_evaluator_rejects_wrong_budget_before_actor_load():
    config = _config(2560)
    bundle = Bundle()
    state = {"training_config": config, "counters": {"env_steps": 32768, "committed_updates": 1280},
             "lora": bundle.adapter_state_dict()}
    with patch("scripts.routea_paired_support.runtime.verify_checkpoint", return_value={
        "lora_identity_hash": "b" * 64, "runtime_identity": {"config_hash": _canonical_hash(config)}}), patch(
        "torch.load", return_value=state):
        with pytest.raises(ValueError, match="budget"):
            load_actor(ROOT, SimpleNamespace(lora_bundle=bundle), {"checkpoint_path": "fixture", "update": 32})
    assert not bundle.loaded
