"""Real E5 runtime checks for the optional X/K sampling protocol."""

import importlib
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch import nn
from diffusers import DDIMScheduler

from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import load_official_symbols


@pytest.fixture
def adapter():
    root = Path(__file__).resolve().parents[3] / "third_party/dppo_official_cc7234a"
    load_official_symbols(root)
    return importlib.import_module("uav_idp3.e5_runtime")


class ConstantX0(nn.Module):
    def forward(self, *, sample, timestep, local_cond, global_cond):
        return torch.full_like(sample, 3.0)


def make_runtime(adapter, **kwargs):
    return adapter.E5PPODiffusion(
        model=ConstantX0(),
        scheduler=DDIMScheduler(num_train_timesteps=100, prediction_type="sample"),
        critic=nn.Linear(576, 1), device="cpu", **kwargs
    )


def draw(runtime, seed=17, deterministic=False):
    torch.manual_seed(seed)
    return runtime({"state": torch.zeros(2, 576), "global_cond": torch.zeros(2, 576)},
                   deterministic=deterministic)


def test_scalar_and_equal_vector_preserve_chain_and_both_logprob_paths(adapter):
    scalar = make_runtime(adapter)
    vector = make_runtime(adapter, min_sampling_denoising_std=[0.1] * 11,
                          min_logprob_denoising_std=[0.1] * 11)
    a, b = draw(scalar), draw(vector)
    assert torch.equal(a.chains, b.chains)
    cond = {"state": torch.zeros(2, 576), "global_cond": torch.zeros(2, 576)}
    assert torch.equal(scalar.get_logprobs(cond, a.chains), vector.get_logprobs(cond, b.chains))
    inds = torch.tensor([0, 9])
    args = (cond, a.chains[:, 0], a.chains[:, 1], inds)
    assert torch.equal(scalar.get_logprobs_subsample(*args), vector.get_logprobs_subsample(*args))
    assert "min_sampling_denoising_std" in dict(vector.named_buffers())
    vector.step()
    assert torch.equal(vector.get_min_sampling_denoising_std(), torch.full((11,), 0.1))


def test_vector_floor_affects_only_z_and_matches_final_gaussian_logprob(adapter):
    floors = [0.1] * 11
    floors[2] = 0.6
    base = make_runtime(adapter)
    raised = make_runtime(adapter, min_sampling_denoising_std=floors,
                          min_logprob_denoising_std=floors)
    a, b = draw(base), draw(raised)
    other = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    assert torch.equal(a.chains[..., other], b.chains[..., other])
    torch.testing.assert_close(b.trajectories[..., 2] - 1.0,
                               6 * (a.trajectories[..., 2] - 1.0), atol=5e-7, rtol=1e-6)
    cond = {"state": torch.zeros(2, 576), "global_cond": torch.zeros(2, 576)}
    lp = raised.get_logprobs_subsample(cond, b.chains[:, -2], b.chains[:, -1],
                                      torch.tensor([9, 9]))
    expected = torch.distributions.Normal(1.0, torch.tensor(floors)).log_prob(b.trajectories)
    assert torch.equal(lp, expected)
    full_lp = raised.get_logprobs(cond, b.chains).reshape(2, 10, 16, 11)
    assert torch.equal(full_lp[:, -1], expected)


def test_explicit_clip_override_preserves_native_check_and_changes_x0(adapter):
    native = make_runtime(adapter)
    parity = make_runtime(adapter, denoised_clip_override=None)
    overridden = make_runtime(adapter, denoised_clip_override=2.0)
    assert torch.equal(draw(native).chains, draw(parity).chains)
    assert torch.equal(draw(native, deterministic=True).trajectories, torch.ones(2, 16, 11))
    assert torch.equal(draw(overridden, deterministic=True).trajectories, torch.full((2, 16, 11), 2.0))
    assert overridden.clip_protocol == {"native": 1.0, "override": 2.0}
    with pytest.raises(adapter.E5RuntimeError, match="native E5 scheduler"):
        make_runtime(adapter, denoised_clip_value=2.0)


def test_scoped_override_restores_runtime_buffers_and_mode_after_exception(adapter):
    runtime = make_runtime(adapter)
    before = draw(runtime, deterministic=True).chains
    state_keys = set(runtime.state_dict())
    with pytest.raises(RuntimeError, match="stop"):
        with adapter.sampling_protocol_override(runtime, min_std_vector=[0.1, 0.1, 0.6] + [0.1] * 8,
                                                denoised_clip_value=2.0, sampling_mode="random") as protocol:
            assert protocol["clip_protocol"]["override"] == 2.0
            assert not torch.equal(draw(runtime, deterministic=True).chains, before)
            raise RuntimeError("stop")
    assert torch.equal(draw(runtime, deterministic=True).chains, before)
    assert set(runtime.state_dict()) == state_keys
    assert runtime.min_sampling_denoising_std == 0.1
    assert runtime.min_logprob_denoising_std == 0.1


def test_override_replaces_historical_scalar_getter_and_restores_exact_methods(adapter):
    runtime = make_runtime(adapter)

    def legacy_scalar_getter(self):
        value = self.min_sampling_denoising_std
        return value if type(value) is float else value()

    runtime.get_min_sampling_denoising_std = MethodType(legacy_scalar_getter, runtime)
    before_getter = runtime.get_min_sampling_denoising_std
    before_forward = runtime.forward
    with adapter.sampling_protocol_override(runtime, min_std_vector=[0.1] * 11):
        assert torch.isfinite(draw(runtime).chains).all()
    assert runtime.get_min_sampling_denoising_std is before_getter
    assert runtime.forward == before_forward


@pytest.mark.parametrize("floor", [[0.1] * 10, [0.1] * 10 + [-0.1], [0.1] * 10 + [float("nan")]])
def test_invalid_vector_floors_fail_before_sampling(adapter, floor):
    with pytest.raises(adapter.E5RuntimeError, match="floor"):
        make_runtime(adapter, min_sampling_denoising_std=floor)
