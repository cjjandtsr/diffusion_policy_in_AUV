"""R1 rework #2 (audit 2026-08-08): CFG λ formal interface lock-down.

The audit found the formal rollout/update adapters accepted an external
``cfg_lambda`` override (``sample_dppo_rollout(..., cfg_lambda=0.0)``) while
the behavior chain is always sampled at ``policy.cfg_lambda`` — the
fake-ratio defect: chain 按 1.5 采样、old_logprob 按 0.0 计算 (a policy at
λ=1.5 + an override 0.0 stores pure-cond logprobs for a CFG-mixed chain).
train_loop never passed an override (the sealed run did not trigger it),
but the public formal interfaces allowed the wrong state.

The ruling:

  - ``sample_dppo_rollout`` / ``make_get_new_logprobs`` / ``_obs_conditions``
    MUST NOT expose a cfg_lambda parameter; the ONLY resolution path is
    ``_require_policy_cfg_lambda`` (the policy attribute), which hard-fails
    on missing (AttributeError), non-finite (ValueError) and anything
    != 1.5 exactly (ValueError)
  - pure-cond contrast experiments call the LOW-LEVEL
    ``per_dim_denoising_logprobs`` / ``per_step_denoising_logp`` directly —
    never the formal adapters with a modified λ
  - ``policy_loader`` verifies the loaded runtime instance has
    ``cfg_lambda == 1.5`` (loader hard-fails on a non-1.5 checkpoint)

Run: pytest -q tests/dppo_parity/test_cfg_contract.py
"""

import inspect
import os

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.chain_logprobs import per_dim_denoising_logprobs
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import EpisodeTaskSpec
from diffusion_policy_3d.rl.dppo_uav.observation_adapter import (
    select_latest_observations,
)
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    _obs_conditions,
    make_get_new_logprobs,
    sample_dppo_rollout,
)

from conftest import DEFAULT_BC_CKPT
from fake_policy import FakePolicy, to_tensor_obs
from mock_uav_env import MockUAVEnv


def _contract() -> RLSystemContract:
    return RLSystemContract()


def _task_specs() -> list:
    return [
        EpisodeTaskSpec(
            world_uid="mock",
            obstacle_json="{}",
            role="anticipatory",
            split="rl_train",
            admission_id="m1",
            generator_version="mock",
            manifest_entry_hash="abc123",
        )
    ]


def _selected(contract: RLSystemContract, seed: int = 0):
    env = MockUAVEnv(contract, seed=seed)
    obs_np, info = env.reset(_task_specs(), seeds=[seed])
    obs_t = to_tensor_obs(obs_np)  # (1, T_h, ...)
    selected, _ = select_latest_observations(obs_t, contract.obs_steps)
    # non-zero quaternion so cond != uncond is observable
    selected["agent_pos"][:, :, 3:7] = 0.5
    return selected


# ═══════════════════════════════════════════════════════════════════
# 1. formal interfaces: no external cfg_lambda (audit P1)
# ═══════════════════════════════════════════════════════════════════

def test_formal_interfaces_do_not_expose_cfg_lambda():
    """Audit P1: the formal rollout/update adapters MUST NOT accept an
    external cfg_lambda — an override there is the fake-ratio defect
    (behavior chain at policy λ, old logprobs at override λ)."""
    for fn in (sample_dppo_rollout, make_get_new_logprobs, _obs_conditions):
        sig = inspect.signature(fn)
        assert "cfg_lambda" not in sig.parameters, (
            f"{fn.__name__} exposes cfg_lambda — audit P1 requires the"
            " formal interface to resolve the λ exclusively from the policy"
        )


def test_policy_lambda_1_5_is_formal_pin():
    """The pinned value: FakePolicy default AND the real phase1a policy
    default are 1.5 — the value the behavior rollout and both likelihood
    paths must run at."""
    contract = _contract()
    assert FakePolicy(contract).cfg_lambda == 1.5
    from diffusion_policy_3d.policy import phase1a_policy as mod

    sig = inspect.signature(mod.Phase1aPolicy.__init__)
    assert sig.parameters["cfg_lambda"].default == 1.5, (
        "formal cfg_lambda pin (1.5) changed — audit contract requires"
        " review before any deviation"
    )


# ═══════════════════════════════════════════════════════════════════
# 2. resolution gate hard-fails (audit P1)
# ═══════════════════════════════════════════════════════════════════

def test_missing_cfg_lambda_hard_fails_rollout():
    """A policy WITHOUT cfg_lambda must fail loudly — the silent getattr
    fallback (pure-cond evaluation) is the pre-rework audited defect."""
    contract = _contract()
    policy = FakePolicy(contract)
    del policy.cfg_lambda  # simulate a policy the attribute was never set on
    with pytest.raises(AttributeError):
        sample_dppo_rollout(policy, _selected(contract), contract)


def test_missing_cfg_lambda_hard_fails_factory():
    """make_get_new_logprobs must resolve the CFG lambda at factory time
    and hard-fail on a missing attribute."""
    contract = _contract()
    policy = FakePolicy(contract)
    del policy.cfg_lambda
    with pytest.raises(AttributeError):
        make_get_new_logprobs(policy, contract)


@pytest.mark.parametrize(
    "bad_lam",
    [float("nan"), float("inf"), 0.0, 2.0],
    ids=["nan", "inf", "zero_pure_cond", "two"],
)
def test_policy_cfg_lambda_outside_1_5_hard_fails_rollout(bad_lam):
    """Non-finite, 0.0 (pure-cond opt-in) and any non-1.5 value hard-fail
    in the rollout — the formal path is pinned to the policy λ == 1.5."""
    contract = _contract()
    policy = FakePolicy(contract)
    policy.cfg_lambda = bad_lam
    with pytest.raises(ValueError):
        sample_dppo_rollout(policy, _selected(contract), contract)


@pytest.mark.parametrize(
    "bad_lam",
    [float("nan"), float("inf"), 0.0, 2.0],
    ids=["nan", "inf", "zero_pure_cond", "two"],
)
def test_policy_cfg_lambda_outside_1_5_hard_fails_factory(bad_lam):
    """Same gate at the update side: make_get_new_logprobs must refuse a
    policy whose λ is not exactly 1.5."""
    contract = _contract()
    policy = FakePolicy(contract)
    policy.cfg_lambda = bad_lam
    with pytest.raises(ValueError):
        make_get_new_logprobs(policy, contract)


# ═══════════════════════════════════════════════════════════════════
# 3. B=1 rollout contract (unchanged)
# ═══════════════════════════════════════════════════════════════════

def test_rollout_batch_greater_than_one_hard_fails():
    """Formal Route A v1: rollout_policy_batch_size == 1 — any B>1 rollout
    hard-fails (never a silent batched rollout)."""
    contract = _contract()
    policy = FakePolicy(contract)
    selected = _selected(contract)
    selected2 = {k: torch.cat([v, v], dim=0) for k, v in selected.items()}
    assert selected2["agent_pos"].shape[0] == 2
    with pytest.raises(ValueError):
        sample_dppo_rollout(policy, selected2, contract)


# ═══════════════════════════════════════════════════════════════════
# 4. CFG mixing is live at λ=1.5
# ═══════════════════════════════════════════════════════════════════

def test_uncond_differs_from_cond_when_cfg():
    """With λ>0 the adapter must produce an uncond condition that exists
    and differs from cond (zeroed quaternion, finite)."""
    contract = _contract()
    policy = FakePolicy(contract)
    selected = _selected(contract)
    cond, uncond = _obs_conditions(policy, selected)
    assert uncond is not None, "λ>0 must produce an uncond condition"
    assert uncond.shape == cond.shape
    assert torch.isfinite(uncond).all()
    assert not torch.allclose(uncond, cond, atol=1e-6), (
        "uncond must differ from cond (zeroed quaternion must change the"
        " encoded condition)"
    )


def test_cfg_mixing_effect_nonzero_finite():
    """The stored logprobs under λ=1.5 differ from a pure-cond recompute on
    the SAME window (mixing wiring is live, not dead) and are finite. The
    pure-cond contrast is computed through the LOW-LEVEL
    per_dim_denoising_logprobs API (cfg_lambda=0.0, no uncond) — the only
    path pure-cond experiments may use after the override removal."""
    contract = _contract()
    policy = FakePolicy(contract)
    selected = _selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    mixed = sample.old_logprobs  # (1, K_f, H_a, 11), λ=1.5
    assert torch.isfinite(mixed).all()
    pure = per_dim_denoising_logprobs(
        chain=sample.chains_policy,
        timesteps=policy.noise_scheduler.timesteps,
        model=policy.model,
        global_cond=sample.obs_condition,
        global_cond_uncond=None,
        cfg_lambda=0.0,
        scheduler=policy.noise_scheduler,
        eta=1.0,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        ft_denoising_steps=contract.ft_denoising_steps,
    )
    assert not torch.allclose(mixed, pure, atol=1e-6), (
        "CFG mixing must change the likelihood (non-zero effect)"
    )


def test_rollout_and_update_use_same_cfg_lambda():
    """make_get_new_logprobs resolves the SAME policy cfg_lambda as the
    rollout, so stored old logprobs reproduce at the trainer's recompute on
    the own chains (≤1e-5) — rollout CFG == update CFG."""
    contract = _contract()
    policy = FakePolicy(contract)
    selected = _selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    get = make_get_new_logprobs(policy, contract)  # no explicit lam
    for d in range(contract.ft_denoising_steps):
        logp = get(
            selected,
            sample.chains_policy[:, d],
            sample.chains_policy[:, d + 1],
            torch.full((1,), d, dtype=torch.long),
        )
        assert torch.isfinite(logp).all()
        assert torch.allclose(
            logp, sample.old_logprobs[:, d], atol=1e-5
        ), f"step {d}: update CFG path diverges from rollout CFG path"


def test_resolved_lambda_shared_across_rollout_and_update(monkeypatch):
    """Spy proof: the behavior sampler AND both likelihood recomputes run
    at the same λ — the rollout old-logprob path (via the adapter) and the
    update new-logprob path (via make_get_new_logprobs) both resolve
    through the single _require_policy_cfg_lambda gate, and the behavior
    chain itself was sampled at the policy's λ."""
    import diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter as adapter_mod

    resolved = []
    orig_resolve = adapter_mod._require_policy_cfg_lambda

    def spy_resolve(policy):
        lam = orig_resolve(policy)
        resolved.append(lam)
        return lam

    sampler_lams = []
    orig_sampler = FakePolicy.sample_ddpm_with_logprob

    def spy_sampler(self, *args, **kwargs):
        sampler_lams.append(self.cfg_lambda)
        return orig_sampler(self, *args, **kwargs)

    monkeypatch.setattr(adapter_mod, "_require_policy_cfg_lambda", spy_resolve)
    monkeypatch.setattr(FakePolicy, "sample_ddpm_with_logprob", spy_sampler)

    contract = _contract()
    policy = FakePolicy(contract)
    selected = _selected(contract)
    sample = sample_dppo_rollout(policy, selected, contract)
    get = make_get_new_logprobs(policy, contract)
    for d in range(contract.ft_denoising_steps):
        get(
            selected,
            sample.chains_policy[:, d],
            sample.chains_policy[:, d + 1],
            torch.full((1,), d, dtype=torch.long),
        )
    assert sampler_lams and all(lam == 1.5 for lam in sampler_lams), (
        "behavior sampler must run at the policy λ (1.5), got"
        f" {sampler_lams}"
    )
    assert resolved and all(lam == 1.5 for lam in resolved), (
        "rollout and update must resolve the same λ (1.5) via the single"
        f" resolution gate, got {resolved}"
    )


# ═══════════════════════════════════════════════════════════════════
# 5. policy_loader verifies the runtime instance λ (audit P1)
# ═══════════════════════════════════════════════════════════════════

def _real_ckpt_dict():
    """The frozen BC checkpoint's raw dict (cfg + state_dicts), loaded on
    CPU once per call — the loader tests must exercise the REAL policy
    construction path (small dims are impossible: iDP3Encoder requires
    point_cloud in shape_meta, so the loader's UNet path is the only
    faithful one). cfg.policy is an OmegaConf DictConfig in struct mode
    (no mutation allowed): shallow-copy it to a plain dict so the tests
    can inject/pop cfg_lambda; the loader only ever .get()s the keys."""
    import torch

    ckpt = torch.load(DEFAULT_BC_CKPT, map_location="cpu", weights_only=False)
    ckpt["cfg"] = dict(ckpt["cfg"])
    ckpt["cfg"]["policy"] = dict(ckpt["cfg"]["policy"])
    return ckpt


@pytest.mark.skipif(
    not os.path.isfile(DEFAULT_BC_CKPT),
    reason="frozen BC checkpoint not present",
)
def test_loader_rejects_non_1_5_ckpt(monkeypatch):
    """policy_loader must hard-fail on a checkpoint whose cfg.policy
    cfg_lambda is not exactly 1.5 — the runtime instance check."""
    import diffusion_policy_3d.rl.dppo_uav.policy_loader as loader_mod

    ckpt = _real_ckpt_dict()
    ckpt["cfg"]["policy"]["cfg_lambda"] = 2.0
    monkeypatch.setattr(
        loader_mod.torch, "load", lambda *args, **kwargs: ckpt
    )
    with pytest.raises(ValueError):
        loader_mod.load_ddpm_policy(DEFAULT_BC_CKPT, device="cpu")


@pytest.mark.skipif(
    not os.path.isfile(DEFAULT_BC_CKPT),
    reason="frozen BC checkpoint not present",
)
def test_loader_missing_cfg_lambda_resolves_1_5(monkeypatch):
    """The frozen BC checkpoint has NO cfg_lambda in cfg.policy — the
    loader's default resolves it to the formal pin 1.5 (production gates
    must not break), and the loaded runtime instance must carry it."""
    import diffusion_policy_3d.rl.dppo_uav.policy_loader as loader_mod

    ckpt = _real_ckpt_dict()
    ckpt["cfg"]["policy"].pop("cfg_lambda", None)
    monkeypatch.setattr(
        loader_mod.torch, "load", lambda *args, **kwargs: ckpt
    )
    policy = loader_mod.load_ddpm_policy(DEFAULT_BC_CKPT, device="cpu")
    assert policy.cfg_lambda == 1.5
