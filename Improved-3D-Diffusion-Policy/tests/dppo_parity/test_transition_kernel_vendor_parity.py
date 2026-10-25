"""K1-K4 + K6: production transition math vs the frozen official DPPO
vendor oracle (review R5.1 / §8.1).

The oracle is the UNMODIFIED vendor DiffusionVPG.p_mean_var (DDIM
branch) — never the project's own formulas. Every DDIM index of a
10-step chain is compared at rtol=1e-5 / atol=1e-6:

  K1 mean; K2 raw/sampling/logprob std (two independent floors);
  K3 per-dim logprob; K4 sample under fixed noise.

The decisive coverage is the LAST index (target = valid timestep 0):
the vendor floors its ~1e-10 raw sigma to min_sampling/min_logprob, so
it is a real stochastic transition — the old production code returned
zero logprob there (F1). K6 asserts the production recompute functions
delegate to the single kernel instead of owning Gaussian/DDIM math.
"""

import pytest
import torch

from diffusion_policy_3d.policy.dppo_transition_kernel import (
    DPPOTransitionKernel,
    build_ddim_tables,
)
from diffusion_policy_3d.rl.dppo_uav.chain_logprobs import (
    per_dim_denoising_logprobs,
)

VENDOR_ROOT = (
    "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
    "/third_party/dppo_official_cc7234a"
)
import sys

if VENDOR_ROOT not in sys.path:
    sys.path.insert(0, VENDOR_ROOT)

from model.diffusion.diffusion_vpg import VPGDiffusion as DiffusionVPG  # noqa: E402

B, R, Da = 2, 8, 11
DDIM_STEPS = 10
FT_STEPS = 5
MIN_SAMPLING_STD = 0.1
MIN_LOGPROB_STD = 0.1
TOL = dict(rtol=1e-5, atol=1e-6)

torch.manual_seed(0)


class _StubEps(torch.nn.Module):
    """Deterministic epsilon oracle: eps = 0.3*sin(x) + 0.1."""

    def forward(self, x, t, cond=None, **kwargs):
        return 0.3 * torch.sin(x) + 0.1


class _StubEta(torch.nn.Module):
    """eta=1 constant network (vendor p_mean_var multiplies per-dim)."""

    def forward(self, cond):
        if isinstance(cond, dict):
            shape = next(iter(cond.values())).shape
        else:
            shape = cond.shape
        return torch.ones(shape[0], Da, device=shape.device if hasattr(shape, "device") else "cpu")


class _StubCritic(torch.nn.Module):
    """Unused by p_mean_var; exists to satisfy the VPG constructor."""

    def forward(self, x, **kwargs):
        return torch.zeros(x.shape[0], 1)


@pytest.fixture(scope="module")
def vendor_model():
    model = DiffusionVPG(
        actor=_StubEps(),
        critic=_StubCritic(),
        ft_denoising_steps=FT_STEPS,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        min_sampling_denoising_std=MIN_SAMPLING_STD,
        min_logprob_denoising_std=MIN_LOGPROB_STD,
        learn_eta=False,
        eta=_StubEta(),
        horizon_steps=R,
        obs_dim=7,
        action_dim=Da,
        device="cpu",
        denoised_clip_value=None,
        eps_clip_value=None,
        denoising_steps=100,
        predict_epsilon=True,
        use_ddim=True,
        ddim_discretize="uniform",
        ddim_steps=DDIM_STEPS,
    )
    model.actor_ft = model.actor  # same stub for both branches
    model.eval()
    return model


def _vendor_transition(model, x_t, index):
    """Vendor mean/std/logprob/sample for one DDIM index (fixed noise)."""
    t_b = torch.full((B,), int(model.ddim_t[index]), dtype=torch.long)
    index_b = torch.full((B,), index, dtype=torch.long)
    cond = {"state": torch.zeros(B, 7)}
    with torch.no_grad():
        mu, logvar, _ = model.p_mean_var(
            x_t, t_b.float(), cond=cond, index=index_b
        )
        raw_std = torch.exp(0.5 * logvar)
        sampling_std = raw_std.clamp(min=MIN_SAMPLING_STD)
        logprob_std = raw_std.clamp(min=MIN_LOGPROB_STD)
        noise = torch.randn(B, R, Da).clamp(-model.randn_clip_value,
                                            model.randn_clip_value)
        x_next = mu + sampling_std * noise
    return {
        "mean": mu,
        "raw_std": raw_std,
        "sampling_std": sampling_std,
        "logprob_std": logprob_std,
        "x_next": x_next,
        "noise": noise,
    }


def _production_kernel(vendor_model):
    tables = build_ddim_tables(
        vendor_model.alphas_cumprod, vendor_model.ddim_t
    )
    return DPPOTransitionKernel(
        tables, ft_denoising_steps=FT_STEPS,
        denoised_clip_value=None, eps_clip_value=None,
    )


def _production_transition(kernel, x_t, index, noise):
    eps = _StubEps()(x_t, None)
    stats = kernel.transition_stats(
        x_t, eps, torch.full((B,), index, dtype=torch.long),
        MIN_SAMPLING_STD, MIN_LOGPROB_STD,
    )
    x_next = kernel.sample(stats, noise)
    return stats, x_next


@pytest.mark.parametrize("index", list(range(DDIM_STEPS)))
def test_k1_k2_mean_and_stds_vendor_parity(vendor_model, index):
    x_t = torch.randn(B, R, Da)
    v = _vendor_transition(vendor_model, x_t, index)
    kernel = _production_kernel(vendor_model)
    stats, x_next = _production_transition(
        kernel, x_t, index, v["noise"]
    )
    assert torch.allclose(stats.mean, v["mean"], **TOL), (
        f"K1 mean parity failed at index {index}"
    )
    assert torch.allclose(stats.raw_std, v["raw_std"], **TOL), (
        f"K2 raw_std parity failed at index {index}"
    )
    assert torch.allclose(stats.sampling_std, v["sampling_std"], **TOL), (
        f"K2 sampling_std parity failed at index {index}"
    )
    assert torch.allclose(stats.logprob_std, v["logprob_std"], **TOL), (
        f"K2 logprob_std parity failed at index {index}"
    )
    assert torch.allclose(x_next, v["x_next"], **TOL), (
        f"K4 sample parity failed at index {index}"
    )
    assert int(stats.optimizer_denoise_index[0].item()) == index - (
        DDIM_STEPS - FT_STEPS
    )
    assert int(stats.scheduler_timestep[0].item()) == int(vendor_model.ddim_t[index])


@pytest.mark.parametrize("index", list(range(DDIM_STEPS)))
def test_k3_per_dim_logprob_vendor_parity(vendor_model, index):
    """Per-dim logprob under the LOGPROB std floor matches the vendor's
    Normal(next_mean, clip(std, min_logprob)).log_prob(chains_next)."""
    from torch.distributions import Normal

    x_t = torch.randn(B, R, Da)
    v = _vendor_transition(vendor_model, x_t, index)
    expected = Normal(
        v["mean"], v["logprob_std"]
    ).log_prob(v["x_next"])
    kernel = _production_kernel(vendor_model)
    stats, _ = _production_transition(
        kernel, x_t, index, v["noise"]
    )
    got = kernel.log_prob(stats, v["x_next"])
    assert got.shape == (B, R, Da), got.shape
    assert torch.allclose(got, expected, **TOL), (
        f"K3 logprob parity failed at index {index}"
    )


def test_k_last_index_is_a_real_stochastic_transition(vendor_model):
    """The transition INTO valid timestep 0 must have floored (non-zero)
    std and non-zero logprob — the old code zeroed it (F1)."""
    kernel = _production_kernel(vendor_model)
    x_t = torch.randn(B, R, Da)
    stats, _ = _production_transition(
        kernel, x_t, DDIM_STEPS - 1, torch.randn(B, R, Da)
    )
    assert float(stats.raw_std.min()) < 1e-3, (
        "last index raw sigma should be ~1e-10 (alpha_prev=1)"
    )
    assert float(stats.sampling_std.min()) == pytest.approx(MIN_SAMPLING_STD)
    assert float(stats.logprob_std.min()) == pytest.approx(MIN_LOGPROB_STD)
    logp = kernel.log_prob(stats, stats.mean + stats.sampling_std * 1e-3)
    assert torch.isfinite(logp).all()
    assert float(logp.abs().max()) > 0.0


def test_k6_production_recompute_delegates_to_official_runtime():
    """The production per-dim recompute must not own Gaussian/DDIM math:
    chain_logprobs / dp_sampler must IMPORT the official runtime facade
    and must NOT re-import the legacy project kernel (R7-2B Step 8; the
    kernel is parity/historical only, docstring mention allowed)."""
    import ast
    import inspect

    from diffusion_policy_3d.rl.dppo_uav import chain_logprobs

    def _imported(tree: ast.AST) -> set:
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                # the imported NAME binds the kernel even when the module
                # string is innocent (`from ...policy import
                # dppo_transition_kernel` — review-fix W1): collect it.
                for a in node.names:
                    names.add(a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.name.split(".")[0])
        return names

    src = inspect.getsource(chain_logprobs)
    imported = _imported(ast.parse(src))
    assert "diffusion_policy_3d.rl.dppo_uav.official_runtime" in imported, (
        "chain_logprobs must delegate to OfficialPhase1aFacade"
    )
    assert not any("dppo_transition_kernel" in m for m in imported), (
        "chain_logprobs must not re-import the legacy project kernel"
    )
    banned = (
        "alpha_bar_t =", "alpha_bar_prev =", "torch.log(2.0 * torch.pi",
        "sigma_t =", "ddim_step_with_logprob",
    )
    for frag in banned:
        assert frag not in src, f"chain_logprobs still owns formula: {frag}"
    dp_mod = __import__("diffusion_policy_3d.policy.dp_sampler",
                        fromlist=[""])
    dp_imported = _imported(ast.parse(inspect.getsource(dp_mod)))
    assert "diffusion_policy_3d.rl.dppo_uav.official_runtime" in dp_imported, (
        "dp_sampler must delegate to OfficialPhase1aFacade"
    )
    assert not any("dppo_transition_kernel" in m for m in dp_imported), (
        "dp_sampler must not re-import the legacy project kernel"
    )
    # Gate self-check (review-fix W1): the natural short form
    # `from diffusion_policy_3d.policy import dppo_transition_kernel`
    # binds the kernel by NAME while its module string is innocent —
    # the collected names above must make the ban predicate fire, or
    # the gate is vacuous again.
    hostile = "from diffusion_policy_3d.policy import dppo_transition_kernel\n"
    hostile_imports = _imported(ast.parse(hostile))
    assert "dppo_transition_kernel" in hostile_imports, (
        "gate self-check: short-form kernel import must be visible"
    )
    assert any("dppo_transition_kernel" in m for m in hostile_imports), (
        "gate self-check: short-form kernel import must be banned"
    )


def test_facade_alpha_one_hard_fails(vendor_model):
    """Independent-review finding 3: a set_alpha_to_one scheduler puts
    alpha == 1.0 into the DDIM tables — the vendor sigma radicand
    (1-alpha_prev)/(1-alpha) becomes 0/0 at the last transition and NaN
    survives clamp. The facade must fail closed, mirroring the legacy
    kernel's alpha==1.0 guard."""
    import copy

    from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
        build_official_phase1a_facade,
    )

    ac = copy.deepcopy(vendor_model.alphas_cumprod)
    ac[0] = 1.0  # set_alpha_to_one=True semantics (ac[timestep 0] = 1)
    scheduler = _facade_scheduler(vendor_model)
    scheduler.alphas_cumprod = ac
    with pytest.raises(ValueError, match="alpha == 1.0"):
        build_official_phase1a_facade(
            model=_StubEpsPhase1a(_StubEps()),
            scheduler=scheduler,
            ft_denoising_steps=FT_STEPS,
            cfg_lambda=0.0,
            device="cpu",
            horizon_steps=R,
            action_dim=Da,
            eta=1.0,
            min_sampling_denoising_std=MIN_SAMPLING_STD,
            min_logprob_denoising_std=MIN_LOGPROB_STD,
        )


def test_facade_ft_exceeds_ddim_steps_hard_fails(vendor_model):
    """Independent-review finding 4: ft_denoising_steps > ddim_steps maps
    step indices to wrong relative timesteps inside the vendor window —
    the vendor constructor asserts ft <= ddim_steps; the facade must
    mirror that assert (fail closed, no silent mis-timestep)."""
    from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
        build_official_phase1a_facade,
    )

    with pytest.raises(ValueError, match="ft_denoising_steps"):
        build_official_phase1a_facade(
            model=_StubEpsPhase1a(_StubEps()),
            scheduler=_facade_scheduler(vendor_model),
            ft_denoising_steps=DDIM_STEPS + 1,
            cfg_lambda=0.0,
            device="cpu",
            horizon_steps=R,
            action_dim=Da,
            eta=1.0,
            min_sampling_denoising_std=MIN_SAMPLING_STD,
            min_logprob_denoising_std=MIN_LOGPROB_STD,
        )


def test_ddim_step_official_clamps_shell_noise_to_randn_clip(vendor_model):
    """Independent-review finding 1: the sampling shell must clamp the
    drawn/injected noise to +-randn_clip_value exactly like the vendor
    forward() (randn_clip_value default 10) and the legacy kernel —
    x_next must use the clamped noise, not the raw draw."""
    from diffusion_policy_3d.policy.dp_sampler import ddim_step_official

    x_t = torch.randn(B, R, Da)
    noisy = torch.randn(B, R, Da) * 4.0  # values well beyond +-10
    assert float(noisy.abs().max()) > 10.0, "test noise must exceed clip"
    with torch.no_grad():
        x_next, _step_logp, facade = ddim_step_official(
            x_t=x_t,
            model=_StubEpsPhase1a(_StubEps()),
            cond=_facade_cond(),
            step_index=DDIM_STEPS - FT_STEPS + 1,
            scheduler=_facade_scheduler(vendor_model),
            cfg_lambda=0.0,
            eta=1.0,
            min_sampling_denoising_std=MIN_SAMPLING_STD,
            min_logprob_denoising_std=MIN_LOGPROB_STD,
            ft_denoising_steps=FT_STEPS,
            noise=noisy,
        )
    assert float(facade.randn_clip_value) == 10.0
    # replicate the vendor transition on the SAME facade
    index_b = torch.full((B,), DDIM_STEPS - FT_STEPS + 1,
                         dtype=torch.long)
    t_b = torch.full((B,), int(vendor_model.ddim_t[
        DDIM_STEPS - FT_STEPS + 1]), dtype=torch.long)
    mean, logvar, _ = facade.p_mean_var(x_t, t_b, _facade_cond(),
                                        index=index_b)
    std = torch.exp(0.5 * logvar).clamp(min=MIN_SAMPLING_STD)
    expected_clamped = mean + std * noisy.clamp(
        -facade.randn_clip_value, facade.randn_clip_value
    )
    expected_unclamped = mean + std * noisy
    assert torch.allclose(x_next, expected_clamped, **TOL), (
        "shell must use noise clamped to +-randn_clip_value"
    )
    assert not torch.allclose(x_next, expected_unclamped, **TOL), (
        "shell must NOT use the raw unclamped noise draw"
    )


# ── Step 10: facade (vendor direct runtime) vs frozen vendor object ──

class _StubEpsPhase1a(torch.nn.Module):
    """Phase1a-signature wrapper over _StubEps.

    The facade actor backend calls the model with keyword args
    (sample/timestep/local_cond/global_cond) — the bare _StubEps
    signature (x, t, cond) would swallow them into **kwargs and raise.
    """

    def __init__(self, stub: torch.nn.Module) -> None:
        super().__init__()
        self.stub = stub

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        return self.stub(sample, None)


def _facade_scheduler(vendor_model):
    """Stub scheduler carrying the vendor's OWN tables — the facade builds
    its DDIM tables from these, so index k reads identical alpha values
    (ddim_alphas_prev layout matches the vendor's flip construction)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        alphas_cumprod=vendor_model.alphas_cumprod,
        timesteps=vendor_model.ddim_t,
        config=SimpleNamespace(prediction_type="epsilon"),
    )


def _build_facade(vendor_model):
    from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
        build_official_phase1a_facade,
    )

    return build_official_phase1a_facade(
        model=_StubEpsPhase1a(_StubEps()),
        scheduler=_facade_scheduler(vendor_model),
        ft_denoising_steps=FT_STEPS,
        cfg_lambda=0.0,
        device="cpu",
        horizon_steps=R,
        action_dim=Da,
        eta=1.0,
        min_sampling_denoising_std=MIN_SAMPLING_STD,
        min_logprob_denoising_std=MIN_LOGPROB_STD,
    )


def _facade_cond():
    from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
        make_official_cond,
    )

    return make_official_cond(torch.zeros(B, 7), None, "cpu")


@pytest.mark.parametrize("index", list(range(DDIM_STEPS)))
def test_step10_p_mean_var_vendor_parity(vendor_model, index):
    """Facade p_mean_var (unbound vendor method over the UAV backends) must
    equal the frozen vendor object's p_mean_var on the SAME tables: mean /
    raw std / sampling std / logprob std / eta (guide Step 10)."""
    facade = _build_facade(vendor_model)
    x_t = torch.randn(B, R, Da)
    t_b = torch.full((B,), int(vendor_model.ddim_t[index]), dtype=torch.long)
    index_b = torch.full((B,), index, dtype=torch.long)
    with torch.no_grad():
        mu_v, logvar_v, eta_v = vendor_model.p_mean_var(
            x_t, t_b.float(), cond={"state": torch.zeros(B, 7)},
            index=index_b,
        )
        mu_f, logvar_f, eta_f = facade.p_mean_var(
            x_t, t_b, cond=_facade_cond(), index=index_b
        )
    raw_v = torch.exp(0.5 * logvar_v)
    raw_f = torch.exp(0.5 * logvar_f)
    assert torch.allclose(mu_f, mu_v, **TOL), (
        f"mean parity failed at index {index}"
    )
    assert torch.allclose(raw_f, raw_v, **TOL), (
        f"raw std parity failed at index {index}"
    )
    assert torch.allclose(
        raw_f.clamp(min=MIN_SAMPLING_STD), raw_v.clamp(min=MIN_SAMPLING_STD),
        **TOL,
    ), f"sampling std parity failed at index {index}"
    assert torch.allclose(
        raw_f.clamp(min=MIN_LOGPROB_STD), raw_v.clamp(min=MIN_LOGPROB_STD),
        **TOL,
    ), f"logprob std parity failed at index {index}"
    assert torch.allclose(eta_f, eta_v, **TOL), (
        f"eta parity failed at index {index}"
    )


@pytest.mark.parametrize("index", list(range(DDIM_STEPS)))
def test_step10_ddim_step_official_sample_logprob_parity(vendor_model, index):
    """The production step (ddim_step_official -> facade -> vendor math)
    must reproduce the frozen vendor object: x_next under the same shell
    noise, per-step logprob 0 outside the FT window and the vendor
    get_logprobs_subsample sum inside it (guide Step 10)."""
    from diffusion_policy_3d.policy.dp_sampler import ddim_step_official

    x_t = torch.randn(B, R, Da)
    v = _vendor_transition(vendor_model, x_t, index)
    x_next, step_logp, _ = ddim_step_official(
        x_t=x_t,
        model=_StubEpsPhase1a(_StubEps()),
        cond=_facade_cond(),
        step_index=index,
        scheduler=_facade_scheduler(vendor_model),
        cfg_lambda=0.0,
        eta=1.0,
        min_sampling_denoising_std=MIN_SAMPLING_STD,
        min_logprob_denoising_std=MIN_LOGPROB_STD,
        ft_denoising_steps=FT_STEPS,
        noise=v["noise"],
    )
    assert torch.allclose(x_next, v["x_next"], **TOL), (
        f"sample parity failed at index {index}"
    )
    if index < DDIM_STEPS - FT_STEPS:
        assert torch.equal(step_logp, torch.zeros(B)), (
            f"base step {index} must carry zero logp (outside FT window)"
        )
    else:
        rel = torch.full(
            (B,), index - (DDIM_STEPS - FT_STEPS), dtype=torch.long
        )
        with torch.no_grad():
            logp_v = vendor_model.get_logprobs_subsample(
                {"state": torch.zeros(B, 7)}, x_t, x_next, rel
            )  # (B, R, Da)
        expected = logp_v.view(B, -1).sum(dim=1)
        assert torch.allclose(step_logp, expected, **TOL), (
            f"step logprob parity failed at index {index}"
        )
        assert torch.isfinite(step_logp).all()


def test_step10_last_index_is_floored_stochastic_via_facade(vendor_model):
    """Guide Step 10: the last transition INTO valid timestep 0 must stay
    a REAL stochastic transition (floored std, non-zero logprob) through
    the facade path — never a zero-logp terminal step."""
    from diffusion_policy_3d.policy.dp_sampler import ddim_step_official

    facade = _build_facade(vendor_model)
    x_t = torch.randn(B, R, Da)
    idx = DDIM_STEPS - 1
    t_b = torch.full((B,), int(vendor_model.ddim_t[idx]), dtype=torch.long)
    index_b = torch.full((B,), idx, dtype=torch.long)
    with torch.no_grad():
        _, logvar, _ = facade.p_mean_var(
            x_t, t_b, cond=_facade_cond(), index=index_b
        )
    raw_std = torch.exp(0.5 * logvar)
    assert float(raw_std.min()) < 1e-3, (
        "last index raw sigma should be ~1e-10 (alpha_prev=1)"
    )
    assert float(
        raw_std.clamp(min=MIN_SAMPLING_STD).min()
    ) == pytest.approx(MIN_SAMPLING_STD)
    _, step_logp, _ = ddim_step_official(
        x_t=x_t,
        model=_StubEpsPhase1a(_StubEps()),
        cond=_facade_cond(),
        step_index=idx,
        scheduler=_facade_scheduler(vendor_model),
        cfg_lambda=0.0,
        eta=1.0,
        min_sampling_denoising_std=MIN_SAMPLING_STD,
        min_logprob_denoising_std=MIN_LOGPROB_STD,
        ft_denoising_steps=FT_STEPS,
        noise=torch.randn(B, R, Da),
    )
    assert torch.isfinite(step_logp).all()
    assert float(step_logp.abs().max()) > 0.0, (
        "last window step (into timestep 0) must carry a floored-std logprob"
    )


def test_production_scheduler_config_all_indices_finite_and_floored():
    """review partA P1-1: the PRODUCTION scheduler config
    (num_train_timesteps=50, squaredcos_cap_v2, prediction_type='sample'
    — the frozen BC checkpoint's stored config) must run the kernel over
    all 10 indices with finite mean/std/logp, the last index floored to
    the std minima, and the sample-mode mean must equal the epsilon-path
    mean on the same tables (Tweedie round-trip equivalence)."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    torch.manual_seed(0)
    sched = DDPMScheduler(
        num_train_timesteps=50,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        # the production loader's scheduler allowlist intentionally drops
        # the checkpoint's stored set_alpha_to_one=True — ac[0] stays < 1
        # and the kernel radicand never hits 0/0 (policy_loader.py:28-34).
        prediction_type="sample",
    )
    sched.set_timesteps(10)
    tables = build_ddim_tables(sched.alphas_cumprod, sched.timesteps)
    kernel = DPPOTransitionKernel(tables, ft_denoising_steps=FT_STEPS)
    from diffusion_policy_3d.policy.dp_sampler import (
        _model_output_to_epsilon,
    )

    x = torch.randn(B, R, Da)
    for index in range(DDIM_STEPS):
        x0 = torch.randn(B, R, Da)  # sample-mode model output (x̂₀)
        # The PRODUCTION conversion (Tweedie, with its 1e-8 guard) —
        # the kernel consumes EPSILON, never raw x̂₀.
        idx_b = torch.full((B,), index, dtype=torch.long)
        a = tables.alpha[index]
        a_prev = tables.alpha_prev[index]
        # EXACT Tweedie (kernel convention: sqrt(1-a) without the 1e-8)
        eps_exact = (x - torch.sqrt(a) * x0) / torch.sqrt(1.0 - a)
        stats = kernel.transition_stats(
            x, eps_exact, idx_b, MIN_SAMPLING_STD, MIN_LOGPROB_STD
        )
        assert torch.isfinite(stats.mean).all(), index
        # manual vendor formula (exact convention) must match the kernel
        sqrt_1m_a = torch.sqrt(1.0 - a)
        x_recon = (x - sqrt_1m_a * eps_exact) / torch.sqrt(a)
        sigma = torch.clamp(
            float(1.0)
            * torch.sqrt(
                (1.0 - a_prev) / (1.0 - a) * (1.0 - a / a_prev)
            ),
            min=1e-10,
        )
        dir_coef = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2,
                                          min=0.0))
        manual_mean = torch.sqrt(a_prev) * x_recon + dir_coef * eps_exact
        assert torch.allclose(stats.mean, manual_mean, **TOL), (
            f"kernel mean diverges from the vendor formula at {index}"
        )
        # The PRODUCTION converter's 1e-8 guard is a documented
        # approximation — pin its bound relative to the exact Tweedie
        # (the last indices amplify it via the small denominator).
        eps_conv = _model_output_to_epsilon(
            x, x0, int(sched.timesteps[index]), sched, "cpu"
        )
        assert torch.allclose(eps_conv, eps_exact, rtol=1e-3, atol=1e-3), (
            f"production Tweedie conversion out of bound at {index}"
        )
        stats_conv = kernel.transition_stats(
            x, eps_conv, idx_b, MIN_SAMPLING_STD, MIN_LOGPROB_STD
        )
        assert torch.isfinite(stats_conv.mean).all(), index
        if index == DDIM_STEPS - 1:
            assert float(stats.sampling_std.min()) == pytest.approx(
                MIN_SAMPLING_STD
            )
            assert float(stats.logprob_std.min()) == pytest.approx(
                MIN_LOGPROB_STD
            )
        logp = kernel.log_prob(stats, stats.mean + stats.sampling_std)
        assert torch.isfinite(logp).all(), index
