"""T5 rework: critic (design §4.13) — obs (3 frames x 192d) -> V(s) (B,).

[ASSUMPTION-9] The T5 "design text not on disk" premise was false; the
design §4.13 specifies the E040 path 3x192 -> mean pool -> 192 -> 128 ->
64 -> 1 (mean-pooled, NOT a 576 concat). The 576 concat is being reworked
to the E040 mean-pool contract (P4); N1 tests here enforce the frozen
encoder / value-head-only optimizer boundary.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import (
    PolicyObsEncoder,
    ValueCritic,
)
from diffusion_policy_3d.rl.dppo_uav.observation_adapter import (
    select_latest_observations,
)

from fake_policy import FakePolicy, to_tensor_obs


def _contract() -> RLSystemContract:
    return RLSystemContract()


def _obs_batch(contract: RLSystemContract, b: int = 2, seed: int = 0):
    rng = np.random.RandomState(seed)
    obs = {
        "point_cloud": rng.uniform(
            -8.0, 8.0, (b, contract.history_steps, contract.point_count, 3)
        ).astype(np.float32),
        "agent_pos": rng.uniform(-1.0, 1.0, (b, contract.history_steps, 9)).astype(
            np.float32
        ),
    }
    return to_tensor_obs(obs)


def _selected(contract: RLSystemContract, b: int = 2, seed: int = 0):
    obs = _obs_batch(contract, b=b, seed=seed)
    selected, _ = select_latest_observations(obs, contract.obs_steps)
    return selected


def _critic(contract: RLSystemContract, policy: FakePolicy) -> ValueCritic:
    return ValueCritic(
        contract=contract, encoder=PolicyObsEncoder(policy)
    )


def test_critic_returns_scalar_per_batch_element():
    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    out = critic(_selected(contract, b=3))
    assert out.shape == (3,), f"expected (B,) got {out.shape}"
    assert torch.isfinite(out).all()


def test_critic_input_dim_is_mean_pooled_192():
    """B1 (ASSUMPTION-9): §4.13 = E040 path 3x192 -> mean pool -> 192 ->
    128 -> 64 -> 1 — the 576 concat is revoked."""
    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    frame_dim = policy.obs_encoder.frame_dim
    first_linear = critic.mlp[1]  # after LayerNorm at index 0
    assert first_linear.in_features == frame_dim, (
        f"expected mean-pooled {frame_dim}, got {first_linear.in_features}"
    )
    assert critic.mlp[3].in_features == 128, (
        "E040 hidden layer must be 128 (was changed to 192 in T5)"
    )


def test_critic_uses_layernorm_and_xavier_like_verified_value_head():
    contract = _contract()
    critic = _critic(contract, FakePolicy(contract))
    assert isinstance(critic.mlp[0], nn.LayerNorm), (
        "E040 ValueHead used LayerNorm for stable training"
    )
    for m in critic.mlp:
        if isinstance(m, nn.Linear):
            w = m.weight.detach()
            std = w.std().item()
            # xavier_uniform_ empirical std: Linear(192,128) ~= 0.079,
            # Linear(64,1) ~= 0.175 — bound must cover both fan shapes
            assert 0.04 < std < 0.2, f"xavier init expected, got std={std}"


def test_critic_backward_does_not_flow_into_encoder():
    """N1 fix (ASSUMPTION-9): critic loss must NOT update the shared BC obs
    encoder — the value head learns on detached features only."""
    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    out = critic(_selected(contract, b=2))
    out.sum().backward()
    grads = [p.grad for p in critic.mlp.parameters() if p.grad is not None]
    assert grads, "no gradients flowed through the value head"
    enc_grads = [
        p.grad for p in policy.obs_encoder.parameters() if p.grad is not None
    ]
    assert not enc_grads, (
        "critic loss must not update the shared policy obs encoder"
    )


def test_critic_optimizer_params_are_value_head_only():
    """N1 fix (ASSUMPTION-9): critic optimizer must contain only the value
    head params — never the shared policy encoder."""
    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    opt_params = set(id(p) for p in critic.value_head_parameters())
    mlp_params = set(id(p) for p in critic.mlp.parameters())
    assert opt_params == mlp_params, (
        "value_head_parameters() must be exactly the value head MLP params"
    )
    policy_params = set(id(p) for p in policy.obs_encoder.parameters())
    assert not (opt_params & policy_params), (
        "critic optimizer params must not include policy params"
    )


def test_policy_obs_encoder_frozen():
    """N1 fix (ASSUMPTION-9): constructing the critic freezes the shared
    obs encoder so no optimizer can ever move it."""
    contract = _contract()
    policy = FakePolicy(contract)
    _critic(contract, policy)
    trainable = [
        p for p in policy.obs_encoder.parameters() if p.requires_grad
    ]
    assert not trainable, "obs encoder params must be frozen after critic construction"


def test_critic_values_differ_across_obs():
    contract = _contract()
    critic = _critic(contract, FakePolicy(contract))
    v1 = critic(_selected(contract, b=2, seed=1))
    v2 = critic(_selected(contract, b=2, seed=2))
    assert not torch.allclose(v1, v2), "critic collapsed to a constant"


def test_critic_mean_pool_permutation_invariance():
    """B1 (ASSUMPTION-9): mean pool is permutation-invariant — swapping the
    frame order must NOT change the value. Recorded as expected behavior
    (temporal sensitivity is a separate ablation only after G6/G7)."""
    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    obs = _obs_batch(contract, b=2, seed=3)
    selected, _ = select_latest_observations(obs, contract.obs_steps)
    # swap the latest and oldest of the selected frames
    flipped = {
        key: torch.flip(value, dims=[1]) for key, value in selected.items()
    }
    v_norm = critic(selected)
    v_flip = critic(flipped)
    assert torch.allclose(v_norm, v_flip, atol=1e-6), (
        "mean-pooled critic must be permutation-invariant over frames"
    )


def test_critic_latest_window_sentinel():
    """B1 (ASSUMPTION-9): the critic consumes only the latest obs_steps
    frames — evaluating on the full history window [0..7] must equal
    evaluating on the latest [5,6,7] after recency selection."""
    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    # full-history batch (history_steps=8 frames per the contract)
    obs = _obs_batch(contract, b=2, seed=4)
    selected, _ = select_latest_observations(obs, contract.obs_steps)
    v_latest = critic(selected)
    # reconstruct the same input from only the latest 3 frames
    latest = {
        key: obs[key][:, -contract.obs_steps:].clone()
        for key in obs
    }
    selected_latest, _ = select_latest_observations(latest, contract.obs_steps)
    v_latest_only = critic(selected_latest)
    assert torch.allclose(v_latest, v_latest_only, atol=1e-6), (
        "value must depend only on the latest observation window"
    )


def test_policy_obs_encoder_output_shape():
    """PolicyObsEncoder: (B, T_o, ...) raw dict -> (B, T_o*192)."""
    contract = _contract()
    policy = FakePolicy(contract)
    enc = PolicyObsEncoder(policy)
    out = enc(_selected(contract, b=2))
    assert out.shape == (2, contract.obs_steps * 192), out.shape


def test_policy_obs_encoder_handles_agent_pos_and_point_cloud():
    """The encoder must consume both keys of the raw obs dict."""
    contract = _contract()
    policy = FakePolicy(contract)
    enc = PolicyObsEncoder(policy)
    out = enc(_selected(contract, b=1))
    assert out.ndim == 2 and out.shape[1] > 0


def test_value_critic_accepts_raw_obs_dict():
    """trainer calls critic(obs_b) with the raw obs dict (T4 contract)."""
    contract = _contract()
    critic = _critic(contract, FakePolicy(contract))
    out = critic(_selected(contract, b=4))
    assert out.shape == (4,)


@pytest.mark.parametrize("b", [1, 5])
def test_critic_batch_sizes(b):
    contract = _contract()
    critic = _critic(contract, FakePolicy(contract))
    assert critic(_selected(contract, b=b)).shape == (b,)


# ═══════════════════════════════════════════════════════════════════
# CRITIC-STATE-ONLY: value-head checkpoint isolation
# ═══════════════════════════════════════════════════════════════════

class _PolicyLike(nn.Module):
    """Minimal nn.Module-shaped policy for the registration-boundary tests.

    FakePolicy is a plain object, so ``self.policy = policy`` inside
    PolicyObsEncoder never registers it as a submodule (nn.Module only
    registers nn.Module children) — the registration leak only exists for
    nn.Module policies (the real Phase1aPolicy). This wrapper gives the
    encoder a real nn.Module to mis-register, so the boundary is exercised.
    """

    def __init__(self, policy: FakePolicy) -> None:
        super().__init__()
        self.normalizer = policy.normalizer
        self.use_pc_color = policy.use_pc_color
        self.n_obs_steps = policy.n_obs_steps
        self.obs_encoder = policy.obs_encoder
        self.model = policy.model


def _policy_like(contract: RLSystemContract) -> _PolicyLike:
    return _PolicyLike(FakePolicy(contract))


def test_policy_obs_encoder_does_not_register_policy():
    """R3.1 (CRITIC-STATE-ONLY): the shared policy must never be registered
    as an nn.Module submodule — a registered reference drags every actor
    parameter (normalizer/obs_encoder/UNet) into the encoder's and critic's
    .parameters() and .state_dict()."""
    enc = PolicyObsEncoder(_policy_like(_contract()))
    assert "policy" not in dict(enc.named_children()), (
        "PolicyObsEncoder must hold the policy by a non-registering "
        "reference, not as an nn.Module submodule"
    )


def test_critic_parameters_disjoint_from_actor_parameters():
    """R3.1 (CRITIC-STATE-ONLY): critic.parameters() must never include
    actor parameters — the checkpoint's value state and the critic optimizer
    both derive from this set."""
    contract = _contract()
    policy_like = _policy_like(contract)
    critic = ValueCritic(contract, encoder=PolicyObsEncoder(policy_like))
    actor_param_ids = {
        id(p)
        for p in list(policy_like.obs_encoder.parameters())
        + list(policy_like.model.parameters())
    }
    critic_param_ids = {id(p) for p in critic.parameters()}
    assert actor_param_ids.isdisjoint(critic_param_ids), (
        "critic.parameters() leaks actor parameters — the value head must "
        "own exactly its own mlp.* params"
    )


def test_value_state_dict_contains_only_value_head():
    """R3.1 (CRITIC-STATE-ONLY): value_state_dict() keys must be exactly
    mlp.* — no policy/normalizer/obs_encoder/UNet/LoRA keys."""
    contract = _contract()
    critic = _critic(contract, FakePolicy(contract))
    keys = set(critic.value_state_dict())
    assert keys, "value state must not be empty"
    assert all(k.startswith("mlp.") for k in keys), (
        f"non-mlp keys in value state: "
        f"{sorted(k for k in keys if not k.startswith('mlp.'))}"
    )
    assert not any("policy" in k or "normalizer" in k for k in keys)


def test_full_critic_state_dict_has_no_policy_keys():
    """R3.1 (CRITIC-STATE-ONLY): even the full critic.state_dict() must not
    embed the policy — the old implementation registered the policy as an
    encoder submodule and value_state embeds the normalizer keys (G6 run-5
    CPU-pollution chain)."""
    contract = _contract()
    critic = ValueCritic(contract, encoder=PolicyObsEncoder(_policy_like(contract)))
    keys = set(critic.state_dict())
    non_mlp = {k for k in keys if not k.startswith("mlp.")}
    assert not non_mlp, (
        f"critic.state_dict() embeds non-value-head keys: {sorted(non_mlp)[:10]}"
    )


def test_load_value_state_dict_roundtrip():
    """R3.1 (CRITIC-STATE-ONLY): value_state_dict -> load_value_state_dict
    restores identical critic outputs and rejects non-mlp payloads."""
    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    obs = _selected(contract, b=2, seed=7)
    v_before = critic(obs)
    critic.load_value_state_dict(critic.value_state_dict())
    v_after = critic(obs)
    assert torch.allclose(v_before, v_after, atol=0.0), (
        "value head must round-trip exactly through value_state_dict"
    )
    with pytest.raises(ValueError):
        critic.load_value_state_dict(
            {"policy.normalizer.scale": torch.zeros(1)}
        )

