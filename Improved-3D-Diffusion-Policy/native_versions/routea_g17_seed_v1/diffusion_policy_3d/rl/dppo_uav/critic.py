"""Value critic (design §4.13) — obs (3 frames x 192d) -> V(s) (B,).

§4.13 contract: the E040 path 3x192 -> mean pool -> 192 -> 128 -> 64 -> 1.
The shared BC obs encoder is frozen at construction and its output
detached, and the critic optimizer must be built from
value_head_parameters() only.
"""

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract


class PolicyObsEncoder(nn.Module):
    """Raw obs dict (B, T_o, ...) -> global_cond (B, T_o*192).

    Mirrors the verified encode_obs_batch path: normalizer.normalize ->
    use_pc_color slice -> per-frame reshape (-1, ...) -> policy.obs_encoder ->
    reshape(B, -1). All downstream callers (rollout, get_new_logprobs,
    critic) route through this one path.

    The encoder wraps the shared BC policy. It must be frozen (freeze())
    so no optimizer can update the actor's observation encoding through
    the critic; actor training touches LoRA params only.

    The policy is held by a NON-REGISTERING reference
    (object.__setattr__ bypasses nn.Module.__setattr__, so the policy is
    never registered as a child submodule) — an equivalent of the
    guide's ``weakref.ref(policy)`` that keeps the encoder self-contained
    without ever exposing the actor in .named_children()/.parameters()/
    .state_dict(). A registered reference (``self.policy = policy``) pulls
    every actor parameter (normalizer/obs_encoder/UNet) into the encoder's
    and critic's .parameters()/.state_dict() — the CPU-pollution chain
    (value_state embedding the normalizer through the registered policy
    submodule) that this design must make impossible.
    """

    def __init__(self, policy) -> None:
        super().__init__()
        # object.__setattr__ skips nn.Module.__setattr__ registration — the
        # strong ref keeps the policy alive without adding it to _modules.
        object.__setattr__(self, "_policy_anchor", policy)

    @property
    def policy(self) -> Any:
        """The non-registered policy reference (encoder-owned lifetime)."""
        return self._policy_anchor

    def freeze(self) -> None:
        """Freeze the shared BC observation encoder.

        The critic forward passes through ``policy.obs_encoder`` — that is
        the shared encoder the critic loss could otherwise update. Freezing
        it here guarantees no optimizer (even one mistakenly built from
        critic.parameters()) can move the actor's observation encoding.
        The UNet is not on the critic forward path and stays untouched
        (actor training is LoRA-only).
        """
        for p in self.parameters():
            p.requires_grad = False
        obs_encoder = getattr(self.policy, "obs_encoder", None)
        params = getattr(obs_encoder, "parameters", None)
        if params is not None:
            for p in params():
                p.requires_grad = False

    def forward(self, obs_b: dict) -> torch.Tensor:
        nobs = self.policy.normalizer.normalize(obs_b)
        if not self.policy.use_pc_color:
            nobs = {**nobs, "point_cloud": nobs["point_cloud"][..., :3]}
        this = {
            key: value[:, : self.policy.n_obs_steps].reshape(
                -1, *value.shape[2:]
            )
            for key, value in nobs.items()
        }
        b = obs_b["agent_pos"].shape[0]
        return self.policy.obs_encoder(this).reshape(b, -1)


class ValueCritic(nn.Module):
    """V(s): (B, T_o, 192) -> mean pool -> (B,) E040-verified head (192->128->64->1).

    §4.13 = the E040 path 3x192 -> mean pool -> 192 -> 128 -> 64 -> 1. Mean
    pooling over the obs_steps frames makes V(s) permutation-invariant over
    the window — temporal sensitivity is a separate ablation.

    The shared BC encoder is frozen at construction and its output detached
    in forward: the critic can never update the actor's observation
    encoding. Build the critic optimizer from value_head_parameters() only.
    """

    def __init__(self, contract: RLSystemContract, encoder: PolicyObsEncoder) -> None:
        super().__init__()
        self.contract = contract
        self.encoder = encoder
        self.encoder.freeze()  # shared BC encoder frozen
        self.frame_dim = getattr(encoder.policy.obs_encoder, "frame_dim", 192)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.frame_dim),
            nn.Linear(self.frame_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def value_head_parameters(self):
        """Value-head-only params for the critic optimizer.

        Never build the critic optimizer from critic.parameters() — that
        set includes the (frozen) shared policy encoder.
        """
        return self.mlp.parameters()

    def value_state_dict(self) -> Dict[str, torch.Tensor]:
        """Value-head-only checkpoint state.

        Keys are exactly ``mlp.*``. The frozen shared encoder carries no
        parameters of its own (the policy is held by a non-registering
        reference, never registered), so this is the complete critic state
        WITHOUT any
        policy/normalizer/obs_encoder/UNet/LoRA keys — the checkpoint's
        value_state can never re-embed or re-move actor state again.
        """
        return {
            k: v for k, v in self.state_dict().items() if k.startswith("mlp.")
        }

    def load_value_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Restore the value head ONLY.

        A payload with any non-``mlp.*`` key is rejected outright (never
        silently dropped) — loading such a state could re-cover the policy
        normalizer (the CPU-pollution chain), which this API must make
        structurally impossible.

        Raises:
            ValueError: payload is empty or contains non-mlp.* keys.
        """
        keys = set(state)
        if not keys:
            raise ValueError("value state must not be empty")
        bad = sorted(k for k in keys if not k.startswith("mlp."))
        if bad:
            raise ValueError(
                "value_state_dict must contain exactly mlp.* keys — "
                f"refusing to load non-value-head keys: {bad[:10]}"
            )
        self.load_state_dict(state, strict=True)

    def forward(self, obs_b: dict) -> torch.Tensor:
        """(B, T_o, ...) raw obs dict -> (B,) value.

        Mean pool the T_o frames before the value head
        (permutation-invariant; temporal sensitivity is a later ablation).
        The encoder runs under no_grad and its output is detached, so the
        value head learns on fixed features and the shared BC encoder is
        never updated by the critic loss.
        """
        with torch.no_grad():
            enc = self.encoder(obs_b)  # (B, T_o*192), frozen
        pooled = enc.reshape(
            -1, self.contract.obs_steps, self.frame_dim
        ).mean(dim=1)
        return self.mlp(pooled.detach()).reshape(-1)
