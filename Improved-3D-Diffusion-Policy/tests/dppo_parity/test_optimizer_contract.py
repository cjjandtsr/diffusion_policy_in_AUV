"""R6-E §11.6.2/§11.6.3 (F-08 → R6-OPT-1, F-09 → R6-DISK-1 E 侧).

§11.6.2: actor and critic each hold EXACTLY one param group for R6/G8.
require_single_param_group is called at creation, after resume and inside
ppo_update; the trainer's clone / grad-norm / finite-check / update-norm
consume the helper's return, never a raw param_groups[0]. A second group
hard-fails BEFORE the first rollout/update — never silently ignored by
[0] indexing. The frozen resolved config (type / lr / betas / eps /
weight_decay / param_group_count / ordered_parameter_name_hash) is
validated per field on resume — a tensor-shape-only comparison is not
enough.

§11.6.3 (fail-before-publish): a non-finite parameter or Adam state
right after optimizer.step() raises the hard exception inside ppo_update;
no partial metrics are returned, no new checkpoint / metrics record is
committed. The full e2e proof runs batch 1 clean (one committed
checkpoint), batch 2 poisoned (raises; the checkpoint file set in
output_dir does not grow). No memory-rollback claim is made — the
transactional boundary is fail-before-publish.

Run: pytest -q tests/dppo_parity/test_optimizer_contract.py
"""

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.optimizer_contract import (
    ResolvedOptimizerConfig,
    ordered_parameter_name_hash,
    require_single_param_group,
    resolved_optimizer_config,
    validate_optimizer_resume_config,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig, ppo_update

from facade_testutil import make_test_context
from test_randperm_epoch_coverage import _marker_buffer  # shared fixture


def _params():
    return [torch.nn.Parameter(torch.zeros(2, 2), requires_grad=True)
            for _ in range(2)]


def _hetero_params():
    """Heterogeneous shapes — required by the order-sensitive tests:
    the hash keys on the cross-process-stable metadata sequence
    (shape/dtype/requires_grad); same-shaped parameters are
    indistinguishable under that view (id() is not resume-stable)."""
    return [
        torch.nn.Parameter(torch.zeros(2, 2), requires_grad=True),
        torch.nn.Parameter(torch.zeros(3, 3), requires_grad=True),
    ]


def _adam(params, lr=1e-3):
    return torch.optim.Adam(params, lr=lr)


# ───────────────────────────────────────────────────────────────────
# §11.6.2 require_single_param_group
# ───────────────────────────────────────────────────────────────────

def test_single_param_group_returns_the_params():
    p = _params()
    opt = _adam(p)
    got = require_single_param_group("actor", opt)
    assert got == p
    assert isinstance(got, list)


def test_second_param_group_hard_fails():
    opt = torch.optim.Adam(
        [{"params": _params()}, {"params": [torch.nn.Parameter(
            torch.zeros(1, requires_grad=True))]}],
        lr=1e-3,
    )
    with pytest.raises(ValueError, match="single param group"):
        require_single_param_group("actor", opt)


def test_empty_param_groups_hard_fails():
    opt = torch.optim.Adam([{"params": []}], lr=1e-3)
    with pytest.raises(ValueError, match="empty|no parameters"):
        require_single_param_group("actor", opt)


# ───────────────────────────────────────────────────────────────────
# §11.6.2 frozen resolved config
# ───────────────────────────────────────────────────────────────────

def test_resolved_config_frozen_fields():
    opt = _adam(_params(), lr=2e-4)
    cfg = resolved_optimizer_config("actor", opt)
    assert isinstance(cfg, ResolvedOptimizerConfig)
    assert cfg.type == "Adam"
    assert cfg.lr == 2e-4
    assert cfg.betas == (0.9, 0.999)
    assert cfg.eps == 1e-8
    assert cfg.weight_decay == 0.0
    assert cfg.param_group_count == 1
    assert len(cfg.ordered_parameter_name_hash) == 64  # sha256 hex


def test_ordered_parameter_name_hash_is_order_sensitive():
    a, b = _hetero_params()
    fwd = ordered_parameter_name_hash([a, b])
    rev = ordered_parameter_name_hash([b, a])
    assert fwd != rev
    assert fwd == ordered_parameter_name_hash([a, b])  # deterministic


def test_resume_identical_config_passes():
    p = _params()
    frozen = resolved_optimizer_config("actor", _adam(p, lr=1e-3))
    # a freshly constructed optimizer with the SAME resolved config
    validate_optimizer_resume_config("actor", _adam(p, lr=1e-3), frozen)


def test_resume_lr_drift_hard_fails():
    p = _params()
    frozen = resolved_optimizer_config("actor", _adam(p, lr=1e-3))
    with pytest.raises(ValueError, match="lr"):
        validate_optimizer_resume_config("actor", _adam(p, lr=1e-4), frozen)


def test_resume_param_order_drift_hard_fails():
    a, b = _hetero_params()
    frozen = resolved_optimizer_config("actor", _adam([a, b]))
    with pytest.raises(ValueError, match="order|hash"):
        validate_optimizer_resume_config("actor", _adam([b, a]), frozen)


def test_resume_second_group_hard_fails():
    a, b = _params()
    frozen = resolved_optimizer_config("actor", _adam([a, b]))
    two_group = torch.optim.Adam(
        [{"params": [a]}, {"params": [b]}], lr=1e-3)
    with pytest.raises(ValueError, match="single param group"):
        validate_optimizer_resume_config("actor", two_group, frozen)


def test_resume_betas_drift_hard_fails():
    p = _params()
    frozen = resolved_optimizer_config("actor", _adam(p))
    drifted = torch.optim.Adam(p, lr=1e-3, betas=(0.9, 0.98))
    with pytest.raises(ValueError, match="betas"):
        validate_optimizer_resume_config("actor", drifted, frozen)


# ───────────────────────────────────────────────────────────────────
# §11.6.2 the trainer consumes the helper — a second group must fail
# BEFORE the first update, not be silently ignored by [0] indexing
# ───────────────────────────────────────────────────────────────────

def _ppo_update_with(actor_optimizer, critic_optimizer,
                     policy, facade, encode_cond):
    contract = RLSystemContract()
    buf = _marker_buffer(contract)
    return ppo_update(
        buffer=buf, contract=contract, facade=facade,
        encode_cond=encode_cond, actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        config=TrainerConfig(
            update_epochs=1, batch_size=8, max_grad_norm=None,
            target_kl=None, ent_coef=0.0, vf_coef=1.0,
        ),
    )


def test_ppo_update_second_param_group_preflight_hard_fails():
    """The pre-change [0] indexing would have trained group 0 only and
    silently ignored the second group — the helper entry check must fail
    before any minibatch."""
    contract = RLSystemContract()
    policy, facade, encode_cond, critic = make_test_context(contract)
    actor_opt = torch.optim.Adam(
        [{"params": list(policy.model.parameters())},
         {"params": [torch.nn.Parameter(torch.zeros(1,
                                                    requires_grad=True))]}],
        lr=1e-3,
    )
    with pytest.raises(ValueError, match="single param group"):
        _ppo_update_with(
            actor_opt, _adam(critic.parameters()), policy, facade,
            encode_cond)


# ───────────────────────────────────────────────────────────────────
# §11.6.3 fail-before-publish: post-step non-finite
# ───────────────────────────────────────────────────────────────────

class _PoisonAfterStep(torch.optim.Adam):
    """Steps normally, then poisons with NaN right after step() — the
    post-step non-finite case §11.6.3 pins (what a diverged step would
    leave in the parameter or the Adam state)."""

    def __init__(self, *args, poison_state=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._poison_state = bool(poison_state)

    def step(self, closure=None):
        super().step(closure)
        p = self.param_groups[0]["params"][0]
        with torch.no_grad():
            if self._poison_state:
                state = self.state.get(p, {})
                assert "exp_avg" in state, "Adam state must exist post-step"
                state["exp_avg"].fill_(float("nan"))
            else:
                p.data.fill_(float("nan"))


def test_post_step_nonfinite_parameter_hard_fails():
    """§11.6.3: a non-finite parameter after optimizer.step() raises the
    hard exception — ppo_update must not return partial metrics.

    The optimizer must wrap the SAME actor ppo_update uses for the
    logprobs recompute (the facade's model) — otherwise no gradient ever
    reaches the optimizer's params and the poison would be vacuous."""
    contract = RLSystemContract()
    policy, facade, encode_cond, critic = make_test_context(contract)
    with pytest.raises(FloatingPointError, match="non-finite"):
        _ppo_update_with(
            _PoisonAfterStep(policy.model.parameters(), lr=1e-3),
            _adam(critic.parameters()),
            policy, facade, encode_cond)


def test_post_step_nonfinite_adam_state_hard_fails():
    """The Adam exp_avg buffer is covered by the same transactional
    boundary as the parameters (same actor/optimizer coupling as above)."""
    contract = RLSystemContract()
    policy, facade, encode_cond, critic = make_test_context(contract)
    with pytest.raises(FloatingPointError, match="non-finite"):
        _ppo_update_with(
            _PoisonAfterStep(policy.model.parameters(), lr=1e-3,
                             poison_state=True),
            _adam(critic.parameters()),
            policy, facade, encode_cond)


def test_poisoned_batch_publishes_no_checkpoint(tmp_path):
    """§11.6.3 end-to-end (production-shaped): batch 1 clean commits one
    checkpoint; batch 2 with a post-step non-finite raises and commits
    NOTHING — the checkpoint file set in output_dir does not grow and no
    metrics.jsonl appears (fail-before-publish; no memory rollback
    claimed)."""
    from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
        OfflineDensity,
    )
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        DEFAULT_ROLE_TARGETS,
        EpisodeTaskSampler,
        EpisodeTaskSpec,
    )
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        LoopConfig,
        run_e2e_batch,
    )

    from fake_policy import FakePolicy
    from mock_uav_env import MockUAVEnv
    from test_train_loop import _contract, _critic, _runtime, _task_specs

    contract = _contract()
    policy = FakePolicy(contract)
    critic = _critic(contract, policy)
    specs = _task_specs()
    env = MockUAVEnv(contract)
    cfg = LoopConfig(
        mode="one_update", n_episodes=2, max_ep_len=12, device="cpu",
        output_dir=str(tmp_path), eval_n_episodes=0,
    )
    runtime = _runtime()

    def drive(actor_opt):
        return run_e2e_batch(
            env=env, policy=policy, critic=critic, contract=contract,
            loop_config=cfg,
            trainer_config=TrainerConfig(device=cfg.device),
            task_specs=specs, actor_optimizer=actor_opt,
            critic_optimizer=_adam(critic.value_head_parameters(), lr=1e-3),
            lora_info={}, runtime=runtime,
            task_sampler=EpisodeTaskSampler(
                specs, DEFAULT_ROLE_TARGETS, base_seed=0),
        )

    # batch 1 clean -> one committed checkpoint
    res = drive(_adam(policy.model.parameters(), lr=1e-3))
    assert res["checkpoint_path"] is not None
    before = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert before, "clean batch must commit a checkpoint"

    # batch 2 poisoned -> raises; the checkpoint set must NOT grow
    with pytest.raises(FloatingPointError, match="non-finite"):
        drive(_PoisonAfterStep(policy.model.parameters(), lr=1e-3))
    after = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert after == before, (
        "poisoned batch must not publish a new checkpoint "
        f"(before={before}, after={after})"
    )
    assert not list(tmp_path.glob("metrics.jsonl")), (
        "poisoned batch must not append committed metrics"
    )
