import pytest
import torch
from types import SimpleNamespace

from scripts.routea_causal_diagnostics.credit_counterfactual import (
    build_credit_variants,
    build_reference_weights,
    compute_credit_variant_gradient,
)
from scripts.dppo_credit_support.credit_probe import (
    compute_full_joint_gradient_reference,
)


def _row(
    transition_id,
    *,
    reward,
    value=0.0,
    next_value=0.0,
    terminated=False,
    truncated=False,
    collection_boundary=False,
    done_reason="none",
    role="dense_recovery",
    episode="ep0",
    decision=0,
    distance_before=2.0,
    distance_after=1.0,
    safe_arrival=False,
    pair_class="stable_failure",
):
    return {
        "transition_id": transition_id,
        "reward": reward,
        "value": value,
        "next_value": next_value,
        "terminated": terminated,
        "truncated": truncated,
        "collection_boundary": collection_boundary,
        "terminal_facts": {"done_reason": done_reason},
        "role": role,
        "episode_identity_hash": episode,
        "decision_index": decision,
        "goal_distance_before_3d": distance_before,
        "goal_distance_after_3d": distance_after,
        "entered_goal_sphere_safe": safe_arrival,
        "pair_class": pair_class,
        "outcome": done_reason,
    }


def test_credit_variants_separate_timeout_bootstrap_from_collection_boundary():
    rows = [
        _row("normal", reward=1.0, next_value=10.0),
        _row(
            "timeout",
            reward=2.0,
            next_value=10.0,
            truncated=True,
            done_reason="time_limit",
            decision=1,
        ),
        _row(
            "boundary",
            reward=3.0,
            next_value=10.0,
            collection_boundary=True,
            episode="ep1",
        ),
        _row(
            "terminated",
            reward=4.0,
            next_value=10.0,
            terminated=True,
            done_reason="collision",
            episode="ep2",
        ),
    ]
    official = torch.tensor([19.9, 11.0, 12.0, 4.0], dtype=torch.float32)

    variants = build_credit_variants(
        rows,
        gamma=0.9,
        gae_lambda=1.0,
        official_advantages=official,
    )

    assert variants["C0-current"]["raw_advantages"] is official
    assert variants["C0-current"]["c0_parity_max_abs"] == 0.0
    assert variants["C1-finite"]["raw_advantages"].tolist() == pytest.approx(
        [11.8, 2.0, 12.0, 4.0]
    )
    assert variants["C2-finite-c10"]["raw_advantages"].tolist() == pytest.approx(
        [2.8, -8.0, 12.0, 4.0]
    )
    assert variants["C3-finite-c20"]["raw_advantages"].tolist() == pytest.approx(
        [-6.2, -18.0, 12.0, 4.0]
    )
    assert torch.equal(
        variants["C3-finite-c20"]["raw_advantages"],
        variants["C4-role-c20"]["raw_advantages"],
    )
    assert variants["C1-finite"]["audit"]["timeout_count"] == 1
    assert variants["C1-finite"]["audit"]["collection_boundary_count"] == 1


def test_c4_normalizes_within_role_without_equal_weighting_roles():
    rows = [
        _row("a0", reward=0.0, role="a", terminated=True, done_reason="reached"),
        _row("a1", reward=1.0, role="a", terminated=True, done_reason="reached"),
        _row("a2", reward=2.0, role="a", terminated=True, done_reason="reached"),
        _row("b0", reward=100.0, role="b", terminated=True, done_reason="reached"),
        _row("b1", reward=102.0, role="b", terminated=True, done_reason="reached"),
    ]
    official = torch.tensor([0.0, 1.0, 2.0, 100.0, 102.0])
    variants = build_credit_variants(
        rows,
        gamma=0.99,
        gae_lambda=0.95,
        official_advantages=official,
    )
    c4 = variants["C4-role-c20"]
    expected = torch.tensor(
        [-1.2041595, 0.0, 1.2041595, -0.9746794, 0.9746794]
    )
    assert torch.allclose(c4["normalized_advantages"], expected, atol=1e-6)
    assert c4["audit"]["normalization"] == "per_role_pair_weighted"
    assert c4["audit"]["role_pair_counts"] == {"a": 3, "b": 2}


def test_credit_variants_fail_closed_on_ambiguous_timeout_and_bad_c0():
    ambiguous = [_row("x", reward=1.0, truncated=True, done_reason="collision")]
    with pytest.raises(ValueError, match="truncated transition must be a real time_limit"):
        build_credit_variants(
            ambiguous,
            gamma=0.99,
            gae_lambda=0.95,
            official_advantages=torch.tensor([1.0]),
        )

    rows = [_row("x", reward=1.0, next_value=2.0)]
    with pytest.raises(ValueError, match="C0 official-path parity"):
        build_credit_variants(
            rows,
            gamma=0.9,
            gae_lambda=0.95,
            official_advantages=torch.tensor([999.0]),
        )


def test_reference_weights_are_positive_mean_scaled_and_do_not_punish_detour():
    rows = [
        _row(
            "t0",
            reward=0.0,
            episode="ep",
            decision=0,
            distance_before=10.0,
            distance_after=8.0,
            pair_class="rescue",
        ),
        _row(
            "t1",
            reward=0.0,
            episode="ep",
            decision=1,
            distance_before=8.0,
            distance_after=9.0,
            pair_class="rescue",
        ),
        _row(
            "t2",
            reward=0.0,
            episode="ep",
            decision=2,
            distance_before=9.0,
            distance_after=7.0,
            safe_arrival=True,
            pair_class="rescue",
            done_reason="reached",
        ),
    ]
    refs = build_reference_weights(rows)
    assert refs["R-progress"]["weights"].tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert refs["R-ratchet"]["weights"].tolist() == pytest.approx(
        [4.0 / 3.0, 0.0, 2.0 / 3.0]
    )
    assert refs["R-safe-arrival"]["weights"].tolist() == [0.0, 0.0, 1.0]
    assert refs["R-rescue"]["weights"].tolist() == [1.0, 1.0, 1.0]
    assert refs["R-reached"]["weights"].tolist() == [0.0, 0.0, 1.0]


class _TinyVariantRuntime:
    gamma_denoising = 1.0
    ft_denoising_steps = 10
    clip_ploss_coef = 0.1
    clip_ploss_coef_base = 0.001
    clip_ploss_coef_rate = 3.0
    clip_advantage_lower_quantile = 0.0
    clip_advantage_upper_quantile = 1.0
    clip_vloss_coef = None

    def __init__(self, weight):
        self.weight = weight

    def get_logprobs_subsample(
        self, condition, chains_prev, chains_next, denoising_inds, get_ent=False
    ):
        coefficients = condition["state"].reshape(-1, 1, 1)
        logprobs = self.weight * coefficients.expand(-1, 16, 11)
        if get_ent:
            return logprobs, torch.zeros(len(logprobs))
        return logprobs

    def critic(self, condition):
        return torch.zeros(len(condition["state"]))


class _TinyVariantBuffer:
    def __init__(self, advantages=None):
        self.roles = ("a", "a", "a", "b", "b")
        self.coefficients = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        self.advantages = (
            torch.zeros(5)
            if advantages is None
            else torch.as_tensor(advantages, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.roles)

    def materialize_pairs(
        self,
        transition_indices,
        denoising_indices,
        *,
        device,
        pass_id,
        mode,
        microbatch_pairs,
        schedule_audit,
    ):
        indices = torch.as_tensor(transition_indices, dtype=torch.long)
        denoising = torch.as_tensor(denoising_indices, dtype=torch.long)
        count = len(indices)
        return SimpleNamespace(
            condition={"state": self.coefficients[indices].to(device)},
            chains_prev=torch.zeros(count, 16, 11, device=device),
            chains_next=torch.zeros(count, 16, 11, device=device),
            denoising_inds=denoising.to(device),
            returns=torch.zeros(count, device=device),
            oldvalues=torch.zeros(count, device=device),
            advantages=self.advantages[indices].to(device),
            oldlogprobs=torch.zeros(count, 16, 11, device=device),
            transition_indices=indices.to(device),
            roles=tuple(self.roles[int(index)] for index in indices),
            pair_count=count,
        )


def test_variant_gradient_uses_explicit_advantages_and_pair_weighted_roles():
    weight = torch.nn.Parameter(torch.tensor(0.0))
    before = weight.detach().clone()
    result = compute_credit_variant_gradient(
        _TinyVariantRuntime(weight),
        _TinyVariantBuffer(),
        (weight,),
        raw_advantages=torch.tensor([0.0, 1.0, 2.0, 100.0, 102.0]),
        roles=("a", "a", "a", "b", "b"),
        group_labels=("t0", "t0", "t1", "t1", "t1"),
        normalization="per_role",
        device=torch.device("cpu"),
        microbatch_pairs=7,
    )
    assert result["role_pair_counts"] == {"a": 30, "b": 20}
    assert result["normalization"] == "per_role"
    assert result["reconstruction_max_abs"] <= 1e-7
    assert torch.allclose(
        result["total_descent"],
        result["role_descent"]["a"] + result["role_descent"]["b"],
        atol=1e-7,
    )
    assert torch.allclose(
        result["total_descent"],
        result["group_descent"]["t0"] + result["group_descent"]["t1"],
        atol=1e-7,
    )
    assert float(result["total_descent"].abs().max()) > 0.0
    assert torch.equal(weight.detach(), before)
    assert weight.grad is None


def test_reference_gradient_does_not_zero_mean_positive_weights():
    weight = torch.nn.Parameter(torch.tensor(0.0))
    result = compute_credit_variant_gradient(
        _TinyVariantRuntime(weight),
        _TinyVariantBuffer(),
        (weight,),
        raw_advantages=torch.ones(5),
        roles=("a", "a", "a", "b", "b"),
        group_labels=("all",) * 5,
        normalization="none",
        device=torch.device("cpu"),
        microbatch_pairs=10,
    )
    assert float(result["total_descent"].abs().max()) > 0.0


class _BatchShapeSensitiveRuntime(_TinyVariantRuntime):
    def get_logprobs_subsample(
        self, condition, chains_prev, chains_next, denoising_inds, get_ent=False
    ):
        coefficients = condition["state"].reshape(-1, 1, 1)
        batch_scale = 1.0 + len(coefficients) * 1e-4
        logprobs = self.weight * batch_scale * coefficients.expand(-1, 16, 11)
        if get_ent:
            return logprobs, torch.zeros(len(logprobs))
        return logprobs


def test_global_variant_matches_official_full_microbatch_before_role_split():
    raw_advantages = torch.tensor([-2.0, -1.0, 0.5, 1.0, 3.0])
    buffer = _TinyVariantBuffer(advantages=raw_advantages)
    weight = torch.nn.Parameter(torch.tensor(0.0))
    runtime = _BatchShapeSensitiveRuntime(weight)

    result = compute_credit_variant_gradient(
        runtime,
        buffer,
        (weight,),
        raw_advantages=raw_advantages,
        roles=buffer.roles,
        group_labels=("all",) * len(buffer),
        normalization="global",
        device=torch.device("cpu"),
        microbatch_pairs=128,
    )
    official_by_role = compute_full_joint_gradient_reference(
        runtime,
        buffer,
        (weight,),
        device=torch.device("cpu"),
    )
    official_descent = -torch.stack(tuple(official_by_role.values())).sum(dim=0)

    assert torch.equal(result["total_descent"], official_descent)
