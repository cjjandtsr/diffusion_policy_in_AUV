from types import SimpleNamespace

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.training_audit import (
    ParameterSnapshot,
    TrainingAudit,
    evaluate_post_update,
    evaluate_transition_credit_audit,
    run_audited_update,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import UpdateMetrics
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import (
    TransitionFacts,
)


REWARD_TERMS = (
    "base",
    "pbrs",
    "onset",
    "future_clearance",
    "trap",
    "flip",
    "support",
    "guard",
)


def _terms(**overrides):
    values = {name: 0.0 for name in REWARD_TERMS}
    values.update(overrides)
    return values


def test_common_task_audit_reports_real_reward_and_space_contract():
    audit = TrainingAudit(term_names=("terminal", "newbest", "risk", "time", "space"),
                          terminate_at_space_limits=False)
    audit.record_transition(task_id="finite", task_role="dense_recovery", reward=-.05,
                            value=0., next_value=0., terminated=False, truncated=True,
                            collection_boundary=True, bootstrap_allowed=False,
                            info={"reward_terms": dict(terminal=0., newbest=0., risk=0., time=-.05, space=0.),
                                  "reward_metrics": {}, "done_reason": "time_limit",
                                  "transition_facts": TransitionFacts.resolve(
                                      collided=False, early_death=False, reached=False, time_limit=True)})
    payload = audit.finalize({"rewards": torch.tensor([-.05]), "values": torch.tensor([0.]),
                              "next_values": torch.tensor([0.]), "advantages": torch.tensor([-.05]),
                              "returns": torch.tensor([-.05])})
    assert payload["reward_schema_version"] == "v2"
    assert payload["altitude_termination_enabled"] is False


def test_training_audit_preserves_reward_episode_and_learning_target_evidence():
    audit = TrainingAudit()
    audit.record_transition(
        task_id="anticipatory-1",
        task_role="anticipatory",
        reward=1.0,
        value=0.25,
        next_value=0.5,
        terminated=True,
        truncated=False,
        collection_boundary=False,
        info={
            "reward_terms": _terms(base=0.75, onset=0.25),
            "reward_metrics": {
                "onset_active": True,
                "risk_gate": 0.8,
                "ignored_vector": [1.0, 2.0],
            },
            "transition_facts": TransitionFacts.resolve(
                collided=False,
                early_death=False,
                reached=True,
                time_limit=False,
            ),
            "done_reason": "reached",
        },
        action_chunk=torch.arange(44, dtype=torch.float32).reshape(4, 11),
    )
    audit.record_transition(
        task_id="dense-1",
        task_role="dense_recovery",
        reward=0.5,
        value=-0.25,
        next_value=0.0,
        terminated=False,
        truncated=False,
        collection_boundary=True,
        info={
            "reward_terms": _terms(pbrs=0.5),
            "reward_metrics": {"onset_active": False, "risk_gate": 0.2},
            "transition_facts": TransitionFacts.resolve(
                collided=False,
                early_death=False,
                reached=False,
                time_limit=False,
            ),
            "done_reason": "none",
        },
        action_chunk=torch.zeros(4, 11),
    )

    payload = audit.finalize(
        {
            "rewards": torch.tensor([1.0, 0.5]),
            "values": torch.tensor([0.25, -0.25]),
            "next_values": torch.tensor([0.5, 0.0]),
            "advantages": torch.tensor([1.5, -0.5]),
            "returns": torch.tensor([1.75, -0.75]),
        }
    )

    assert payload["transitions"] == 2
    assert payload["reward"]["total"]["sum"] == pytest.approx(1.5)
    assert payload["reward"]["terms"]["onset"]["sum"] == pytest.approx(0.25)
    assert payload["reward"]["terms_by_role"]["anticipatory"]["onset"][
        "sum"
    ] == pytest.approx(0.25)
    assert payload["reward"]["terms_by_role"]["dense_recovery"]["pbrs"][
        "sum"
    ] == pytest.approx(0.5)
    assert payload["reward"]["term_sum_error_max_abs"] == pytest.approx(0.0)
    assert payload["episodes"]["completed"] == 1
    assert payload["episodes"]["partial_segments"] == 1
    assert payload["episodes"]["done_reason_counts"] == {
        "collection_boundary": 1,
        "reached": 1,
    }
    assert payload["episodes"]["role_segment_counts"] == {
        "anticipatory": 1,
        "dense_recovery": 1,
    }
    assert payload["reward_metrics"]["boolean_true_counts"]["onset_active"] == 1
    assert payload["reward_metrics"]["boolean_true_rates"]["onset_active"] == pytest.approx(
        0.5
    )
    assert payload["reward_metrics"]["numeric"]["risk_gate"]["mean"] == pytest.approx(
        0.5
    )
    assert payload["reward_metrics"]["ignored_non_scalar_keys"] == [
        "ignored_vector"
    ]
    assert payload["reward_metrics"]["by_role"]["anticipatory"][
        "boolean_true_rates"
    ]["onset_active"] == pytest.approx(1.0)
    assert payload["reward_metrics"]["by_role"]["dense_recovery"]["numeric"][
        "risk_gate"
    ]["mean"] == pytest.approx(0.2)
    assert payload["learning_targets"]["advantages"]["mean"] == pytest.approx(0.5)
    assert payload["learning_targets"]["advantages"]["p05"] == pytest.approx(-0.4)
    assert payload["learning_targets"]["advantages"]["p50"] == pytest.approx(0.5)
    assert payload["learning_targets"]["advantages"]["p95"] == pytest.approx(1.4)
    assert payload["learning_targets"]["advantages"]["nonzero_count"] == 2
    assert payload["learning_targets"]["advantages"]["nonzero_rate"] == pytest.approx(1.0)
    assert payload["learning_targets"]["returns"]["min"] == pytest.approx(-0.75)
    assert payload["learning_targets_by_role"]["anticipatory"]["advantages"][
        "mean"
    ] == pytest.approx(1.5)
    assert payload["learning_targets_by_role"]["dense_recovery"]["advantages"][
        "mean"
    ] == pytest.approx(-0.5)
    assert payload["critic"]["explained_variance"] == pytest.approx(0.36)
    assert payload["action"]["rollout_view_shape"] == [4, 11]
    assert payload["action"]["per_dimension"]["0"]["count"] == 8
    assert payload["action"]["by_role"]["anticipatory"]["per_dimension"]["0"][
        "max"
    ] == pytest.approx(33.0)
    assert payload["episodes"]["completed_return_by_role"] == {
        "anticipatory": {
            "count": 1,
            "sum": pytest.approx(1.0),
            "mean": pytest.approx(1.0),
            "std": pytest.approx(0.0),
            "min": pytest.approx(1.0),
            "max": pytest.approx(1.0),
            "p05": pytest.approx(1.0),
            "p50": pytest.approx(1.0),
            "p95": pytest.approx(1.0),
            "nonzero_count": 1,
            "nonzero_rate": pytest.approx(1.0),
        }
    }
    assert payload["episodes"]["done_reason_counts_by_role"] == {
        "anticipatory": {"reached": 1}
    }
    assert payload["episodes"]["role_transition_counts"] == {
        "anticipatory": 1,
        "dense_recovery": 1,
    }
    assert payload["episodes"]["completed_length_by_role"]["anticipatory"][
        "mean"
    ] == pytest.approx(1.0)


def test_training_audit_rejects_missing_reward_terms():
    audit = TrainingAudit()
    with pytest.raises(ValueError, match="reward term keys"):
        audit.record_transition(
            task_id="task",
            task_role="anticipatory",
            reward=0.0,
            value=0.0,
            next_value=0.0,
            terminated=False,
            truncated=False,
            collection_boundary=True,
            info={"reward_terms": {"base": 0.0}, "reward_metrics": {}},
        )


def test_v17_audit_requires_identity_context_and_binds_learning_targets():
    from diffusion_policy_3d.env.reward_v17 import TERM_NAMES

    audit = TrainingAudit(term_names=TERM_NAMES)
    terms = {name: 0.0 for name in TERM_NAMES}
    info = {
        "reward_terms": terms,
        "reward_metrics": {},
        "transition_facts": TransitionFacts.resolve(reached=False),
        "done_reason": "none",
    }
    with pytest.raises(ValueError, match="requires audit_context"):
        audit.record_transition(
            task_id="task", task_role="anticipatory", reward=0.0,
            value=0.0, next_value=0.0, terminated=False, truncated=False,
            collection_boundary=True, info=info,
        )

    sha = "a" * 64
    audit.record_transition(
        task_id="task", task_role="anticipatory", reward=0.0,
        value=0.0, next_value=0.0, terminated=False, truncated=False,
        collection_boundary=True, info=info, action_chunk=torch.zeros(4, 11),
        audit_context={
            "run_seed": 1, "update_index": 1, "policy_state_update_index": 0,
            "transition_id": "u1:t0", "episode_id": sha, "decision_index": 7,
            "task_identity_hash": sha, "world_id": "world", "task_seed": 2,
            "reset_seed": 3, "admission_manifest_hash": sha,
            "runtime_identity_hash": sha, "source_set_hash": sha,
            "config_hash": sha, "checkpoint_identity_hash": sha,
            "goal_xyz": [1.0, 0.0, 1.0],
            "position_before_xyz": [0.0, 0.0, 1.0],
            "position_after_xyz": [0.5, 0.0, 1.0],
            "goal_distance_before_3d": 1.0, "goal_distance_after_3d": 0.5,
            "goal_error_xy_before": 1.0, "goal_error_z_before": 0.0,
            "goal_error_xy_after": 0.5, "goal_error_z_after": 0.0,
            "observation_provenance": {"policy_input_hash": sha},
            "observation_provenance_after": {"policy_input_hash": sha},
        },
    )
    payload = audit.finalize({
        "rewards": torch.tensor([0.0]), "values": torch.tensor([0.0]),
        "next_values": torch.tensor([0.0]), "advantages": torch.tensor([2.0]),
        "returns": torch.tensor([2.0]),
    })
    record = payload["transition_records"][0]
    assert payload["schema_version"] == 5
    assert record["advantage_raw"] == pytest.approx(2.0)
    assert record["advantage_normalized_outer_update"] == pytest.approx(0.0)
    assert record["return_target"] == pytest.approx(2.0)


def test_parameter_snapshot_reports_nonzero_actor_and_critic_deltas():
    actor = [torch.nn.Parameter(torch.tensor([1.0, 2.0]))]
    critic = [torch.nn.Parameter(torch.tensor([-1.0, 3.0]))]
    snapshot = ParameterSnapshot.capture(actor=actor, critic=critic)
    with torch.no_grad():
        actor[0][0].add_(0.5)
        critic[0].add_(torch.tensor([0.0, -0.25]))

    delta = snapshot.measure(actor=actor, critic=critic)

    assert delta["actor"]["l2_norm"] == pytest.approx(0.5)
    assert delta["actor"]["max_abs"] == pytest.approx(0.5)
    assert delta["actor"]["nonzero_elements"] == 1
    assert delta["critic"]["l2_norm"] == pytest.approx(0.25)
    assert delta["critic"]["nonzero_elements"] == 1


def test_evaluate_post_update_recomputes_official_metrics_without_gradients():
    class Runtime:
        critic = torch.nn.Linear(1, 1)

        def loss(self, *args, **kwargs):
            assert not torch.is_grad_enabled()
            torch.rand(())
            assert kwargs["reward_start"] == 2
            assert kwargs["reward_horizon"] == 4
            return (
                torch.tensor(0.1),
                torch.tensor(-1.0),
                torch.tensor(2.0),
                0.25,
                0.03,
                1.1,
                torch.tensor(0.0),
                torch.tensor(0.0),
            )

    runtime = Runtime()
    trainer = SimpleNamespace(
        runtime=runtime,
        critic_parameters=tuple(runtime.critic.parameters()),
        config=SimpleNamespace(
            reward_start=2,
            reward_horizon=4,
            value_loss_coefficient=0.5,
            entropy_loss_coefficient=0.0,
        ),
    )
    batch = SimpleNamespace(
        condition={},
        chains_prev=torch.zeros(1),
        chains_next=torch.zeros(1),
        denoising_inds=torch.zeros(1),
        returns=torch.zeros(1),
        oldvalues=torch.zeros(1),
        advantages=torch.zeros(1),
        oldlogprobs=torch.zeros(1),
    )

    rng_before = torch.get_rng_state().clone()
    metrics = evaluate_post_update(trainer, batch)

    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert metrics == {
        "policy_loss": pytest.approx(0.1),
        "value_loss": pytest.approx(2.0),
        "entropy_loss": pytest.approx(-1.0),
        "total_loss": pytest.approx(1.1),
        "approx_kl": pytest.approx(0.03),
        "ratio_mean": pytest.approx(1.1),
        "clip_fraction": pytest.approx(0.25),
    }


def test_run_audited_update_separates_pre_post_and_parameter_delta():
    actor = torch.nn.Parameter(torch.tensor([0.0]))
    critic = torch.nn.Parameter(torch.tensor([0.0]))

    class Runtime:
        critic = SimpleNamespace(parameters=lambda: (critic,))

        def loss(self, *args, **kwargs):
            return (
                torch.tensor(0.2),
                torch.tensor(-1.0),
                torch.tensor(1.0),
                0.1,
                0.02,
                1.05,
                torch.tensor(0.0),
                torch.tensor(0.0),
            )

    class Trainer:
        actor_parameters = (actor,)
        critic_parameters = (critic,)
        runtime = Runtime()
        config = SimpleNamespace(
            reward_start=2,
            reward_horizon=4,
            value_loss_coefficient=0.5,
            entropy_loss_coefficient=0.0,
        )

        def update(self, batch, *, update_actor=True):
            with torch.no_grad():
                if update_actor:
                    actor.add_(0.1)
                critic.add_(-0.2)
            return UpdateMetrics(
                0.0,
                2.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.5 if update_actor else 0.0,
                3.0,
                actor_updated=update_actor,
            )

    batch = SimpleNamespace(
        condition={},
        chains_prev=torch.zeros(1),
        chains_next=torch.zeros(1),
        denoising_inds=torch.zeros(1),
        returns=torch.zeros(1),
        oldvalues=torch.zeros(1),
        advantages=torch.zeros(1),
        oldlogprobs=torch.zeros(1),
    )

    evidence = run_audited_update(Trainer(), batch, update_actor=False)

    assert set(evidence) == {"pre_update", "post_update", "parameter_delta"}
    assert evidence["pre_update"]["actor_grad_norm"] == pytest.approx(0.0)
    assert evidence["pre_update"]["actor_updated"] is False
    assert evidence["post_update"]["approx_kl"] == pytest.approx(0.02)
    assert evidence["parameter_delta"]["actor"]["l2_norm"] == pytest.approx(0.0)
    assert evidence["parameter_delta"]["critic"]["l2_norm"] == pytest.approx(0.2)

    skipped = run_audited_update(
        Trainer(), batch, update_actor=False, evaluate_after=False
    )
    assert skipped["post_update"] is None


def test_transition_credit_audit_binds_post_update_credit_to_each_transition():
    class Runtime:
        gamma_denoising = 0.8
        ft_denoising_steps = 10
        clip_ploss_coef_base = 0.001
        clip_ploss_coef = 0.1
        clip_ploss_coef_rate = 3.0
        clip_vloss_coef = None
        min_logprob_denoising_std = 0.1
        ddim_steps = 10
        ddim_t = torch.arange(10)

        @staticmethod
        def critic(condition):
            return torch.zeros(condition["state"].shape[0])

        @staticmethod
        def get_logprobs_subsample(
            condition, chains_prev, chains_next, denoising_inds, *, get_ent
        ):
            assert get_ent is True
            return torch.full_like(chains_next, 0.2), torch.zeros(
                chains_next.shape[0], device=chains_next.device
            )

        @staticmethod
        def p_mean_var(
            chains_prev, timesteps, *, cond, index, use_base_policy
        ):
            mean = torch.zeros_like(chains_prev)
            if not use_base_policy:
                mean = mean + 0.1
            log_variance = torch.zeros(
                chains_prev.shape[0],
                1,
                chains_prev.shape[-1],
                device=chains_prev.device,
            )
            return mean, log_variance, torch.zeros(
                chains_prev.shape[0], device=chains_prev.device
            )

    class Buffer:
        _items = (
            {"transition_id": "u1:t0"},
            {"transition_id": "u1:t1"},
        )

        def __len__(self):
            return 2

        @staticmethod
        def materialize_pairs(
            transition_indices, denoising_indices, *, device, **unused
        ):
            count = len(transition_indices)
            return SimpleNamespace(
                condition={"state": torch.zeros(count, 3, device=device)},
                chains_prev=torch.zeros(count, 16, 11, device=device),
                chains_next=torch.zeros(count, 16, 11, device=device),
                denoising_inds=denoising_indices.to(device),
                returns=torch.ones(count, device=device),
                oldvalues=torch.zeros(count, device=device),
                advantages=torch.tensor([1.0, -1.0], device=device),
                oldlogprobs=torch.zeros(count, 16, 11, device=device),
                transition_indices=transition_indices.to(device),
                transition_ids=("u1:t0", "u1:t1"),
                roles=("anticipatory", "dense_recovery"),
                outcomes=("none", "collision"),
                provenance_hashes=("a" * 64, "b" * 64),
            )

    trainer = SimpleNamespace(
        runtime=Runtime(),
        critic_parameters=(torch.nn.Parameter(torch.zeros(1)),),
        config=SimpleNamespace(reward_start=2, reward_horizon=4),
    )
    records = evaluate_transition_credit_audit(
        trainer, Buffer(), microbatch_pairs=1
    )

    assert [record["transition_id"] for record in records] == ["u1:t0", "u1:t1"]
    assert all(0 <= record["denoising_step_k"] < 10 for record in records)
    assert all(record["old_log_prob"] == pytest.approx(0.0) for record in records)
    assert all(record["new_log_prob"] == pytest.approx(0.2) for record in records)
    assert all(record["ratio"] == pytest.approx(torch.exp(torch.tensor(0.2)).item()) for record in records)
    assert all(record["analytic_bc_reverse_kl_sum"] > 0.0 for record in records)
    assert all(record["observation_provenance_hash"] in {"a" * 64, "b" * 64} for record in records)
