from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.identity import (
    IdentityMismatch,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import (
    restore_checkpoint,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.save import save_checkpoint
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import (
    CheckpointHeader,
    SCHEMA_VERSION,
)


class _LoRA:
    def __init__(self):
        self.value = nn.Parameter(torch.tensor(2.0))

    def adapter_state_dict(self):
        return {"value": self.value.detach().clone()}

    def load_adapter_state_dict(self, state):
        with torch.no_grad():
            self.value.copy_(state["value"])


def _components():
    lora = _LoRA()
    critic = nn.Linear(2, 1)
    actor_optimizer = torch.optim.Adam([lora.value], lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    counters = SimpleNamespace(attempted_updates=3, committed_updates=2)
    return lora, critic, actor_optimizer, critic_optimizer, counters


def test_identity_mismatch_happens_before_tensor_load(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    header = CheckpointHeader(SCHEMA_VERSION, {}, "a" * 64, "b" * 64, 2)
    (root / "header.json").write_text(header.to_json(), encoding="utf-8")
    called = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: called.append(True))
    lora, critic, actor_opt, critic_opt, counters = _components()
    with pytest.raises(IdentityMismatch):
        restore_checkpoint(
            root,
            runtime_identity=SimpleNamespace(content_hash="c" * 64),
            lora_identity=SimpleNamespace(content_hash="b" * 64),
            lora_bundle=lora,
            critic=critic,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
            counters=counters,
        )
    assert called == []


def test_checkpoint_roundtrip_restores_parameters_counters_and_rng(tmp_path):
    lora, critic, actor_opt, critic_opt, counters = _components()
    runtime_identity = SimpleNamespace(content_hash="a" * 64)
    lora_identity = SimpleNamespace(content_hash="b" * 64)
    header = CheckpointHeader(SCHEMA_VERSION, {}, "a" * 64, "b" * 64, 2)
    torch.manual_seed(77)
    save_checkpoint(
        tmp_path / "checkpoint",
        header=header,
        lora_bundle=lora,
        critic=critic,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        counters=counters,
    )
    expected_random = torch.rand(4)
    with torch.no_grad():
        lora.value.zero_()
        for parameter in critic.parameters():
            parameter.zero_()
    counters.committed_updates = 99
    restore_checkpoint(
        tmp_path / "checkpoint",
        runtime_identity=runtime_identity,
        lora_identity=lora_identity,
        lora_bundle=lora,
        critic=critic,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        counters=counters,
    )
    torch.testing.assert_close(lora.value, torch.tensor(2.0))
    assert counters.committed_updates == 2
    torch.testing.assert_close(torch.rand(4), expected_random)

    if torch.cuda.is_available():
        restore_checkpoint(
            tmp_path / "checkpoint",
            runtime_identity=runtime_identity,
            lora_identity=lora_identity,
            lora_bundle=lora,
            critic=critic,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
            counters=counters,
            map_location="cuda:0",
        )
        torch.testing.assert_close(torch.rand(4), expected_random)
