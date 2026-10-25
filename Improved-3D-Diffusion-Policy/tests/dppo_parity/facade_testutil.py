"""R7-3 shared CPU test context: the OFFICIAL facade over FakePolicy.

ppo_update's signature changed with R7-3 (guide Task R7-3 Step 3): the
B=1 likelihood microbatch loop is deleted and the per-minibatch loss is
a SINGLE official ``facade.loss`` call at the actual minibatch batch.
The old test doubles (FakeActor / FakeCritic / get_new_logprobs spies)
cannot exercise that path, so every CPU test that drives ppo_update now
runs the PRODUCTION entry: the real ``OfficialPhase1aFacade`` (vendor
PPODiffusion.loss math) over the FakePolicy model, the canonical
``_obs_conditions`` + ``make_official_cond`` encode path, and a
cond-interface critic for the vendor value loss.

The facade's scheduler is a fresh DDPMScheduler WITHOUT the
``set_alpha_to_one`` flag — diffusers 0.38.0 removed the argument, so
the default behavior already drops it (alphas_cumprod[-1] stays ~4e-5);
the real loader never passes the flag either
(policy_loader._VALID_SCHEDULER_KEYS, the 0.38 key set), and the facade
hard-fails on an alpha == 1.0 DDIM table entry
(official_runtime.py:282-287). The fake policy's OWN scheduler keeps
its default config (other tests pin its sampling math); the facade only
reads its own scheduler's tables.

Run: imported by the CPU ppo_update tests; nothing to run directly.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
    OfficialPhase1aFacade,
    build_official_phase1a_facade,
    make_official_cond,
)
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    _obs_conditions,
)

from fake_policy import FakePolicy


def make_test_scheduler(contract: RLSystemContract) -> DDPMScheduler:
    """Scheduler mirroring the real loader's config (diffusers 0.38:
    no set_alpha_to_one argument — the default already avoids the
    alpha==1.0 alphas_cumprod entry the facade hard-fails on)."""
    sched = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    sched.set_timesteps(contract.denoising_steps)
    return sched


class CondFakeCritic(nn.Module):
    """Cond-interface critic for the vendor value loss.

    The vendor ``PPODiffusion.loss`` calls ``self.critic(obs).view(-1)``
    with the flat cond dict (``global_cond`` (B, To*192)). Mirrors
    OfficialValueHead's pooling (mean over the obs steps) and its
    ``pooled.detach()`` (the encoder must never be updated through the
    value head; the critic optimizer owns only the mlp).
    """

    def __init__(
        self,
        contract: RLSystemContract,
        obs_steps: int = 3,
        frame_dim: int = 192,
    ) -> None:
        super().__init__()
        self.contract = contract
        self._obs_steps = obs_steps
        self._frame_dim = frame_dim
        self.mlp = nn.Linear(frame_dim, 1)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, obs_cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        gc = obs_cond["global_cond"]  # (B, To*D)
        b = gc.shape[0]
        pooled = gc.reshape(b, self._obs_steps, self._frame_dim).mean(
            dim=1
        ).detach()
        return self.mlp(pooled).reshape(-1)


def make_test_facade(
    policy: FakePolicy,
    contract: RLSystemContract,
    critic: Optional[nn.Module] = None,
    device: str = "cpu",
    min_logprob_denoising_std: float = 0.1,
    cfg_lambda: Optional[float] = None,
) -> OfficialPhase1aFacade:
    """Official facade over the fake policy's model (R7-3 production path).

    Mirrors the train_loop one_update wiring (guide R7-3 Step 3/6):
    same builder arguments, eta=1.0, contract hyperparameters, and the
    policy's formal CFG lambda (1.5). ``cfg_lambda`` is the explicit
    pure-cond CONTRAST override (0.0) — the facade constructor is the
    only place a diagnostic lambda may enter; the canonical adapter path
    stays pinned (phase1a_actor_adapter._require_policy_cfg_lambda).
    """
    return build_official_phase1a_facade(
        model=policy.model,
        scheduler=make_test_scheduler(contract),
        ft_denoising_steps=contract.ft_denoising_steps,
        cfg_lambda=(
            policy.cfg_lambda if cfg_lambda is None else cfg_lambda
        ),
        device=device,
        horizon_steps=contract.action_horizon,
        action_dim=contract.action_dim,
        eta=1.0,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=min_logprob_denoising_std,
        critic=critic,
        gamma_denoising=contract.gamma_denoising,
        norm_adv=True,
        randn_clip_value=10.0,
    )


def make_test_encode_cond(
    policy: FakePolicy, device: str = "cpu"
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Canonical encode_cond: _obs_conditions -> official cond dict.

    Identical to the train_loop one_update wiring — the CFG uncond
    branch is built whenever the policy's lambda > 0 (formal 1.5).
    """

    def encode_cond(obs_b: Dict[str, torch.Tensor]):
        cond, uncond = _obs_conditions(policy, obs_b)
        return make_official_cond(cond, uncond, device)

    return encode_cond


def make_test_context(
    contract: RLSystemContract,
    critic: Optional[nn.Module] = None,
    device: str = "cpu",
    min_logprob_denoising_std: float = 0.1,
) -> Tuple[
    FakePolicy,
    OfficialPhase1aFacade,
    Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    nn.Module,
]:
    """Full CPU test context: (policy, facade, encode_cond, critic).

    The actor optimizer must be built over ``policy.model.parameters()``
    (the facade's backends wrap exactly that model); the critic
    optimizer over the returned critic's parameters (CondFakeCritic.mlp
    by default).
    """
    policy = FakePolicy(contract)
    critic = critic or CondFakeCritic(contract)
    facade = make_test_facade(
        policy, contract, critic=critic, device=device,
        min_logprob_denoising_std=min_logprob_denoising_std,
    )
    return policy, facade, make_test_encode_cond(policy, device), critic
