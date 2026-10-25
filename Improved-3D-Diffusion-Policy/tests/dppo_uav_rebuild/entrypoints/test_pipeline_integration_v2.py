"""Cross-module ordering and audit-consumption contract for the production loop."""

import inspect

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import pipeline


def test_production_transition_consumption_order_is_fail_closed():
    source = inspect.getsource(pipeline._run_unlocked)
    ordered_markers = (
        "build_runtime_identity(",
        "task_sampler.sample()",
        "adapter.reset(",
        "system.observation_adapter.encode(",
        "system.observation_adapter.verify(encoded_observation)",
        "system.critic(critic_condition)",
        "system.rollout_sampler.sample(",
        "adapter.step(",
        "audit.record_transition(",
        "buffer.add(",
    )
    offsets = [source.index(marker) for marker in ordered_markers]
    assert offsets == sorted(offsets)


def test_pipeline_consumes_full_v5_identity_and_endpoint_context():
    source = inspect.getsource(pipeline._run_unlocked)
    required = (
        '"policy_state_update_index"', '"episode_id"', '"decision_index"',
        '"task_identity_hash"', '"reset_seed"', '"admission_manifest_hash"',
        '"runtime_identity_hash"', '"source_set_hash"', '"config_hash"',
        '"checkpoint_identity_hash"', '"goal_distance_before_3d"',
        '"goal_distance_after_3d"', '"goal_error_xy_before"',
        '"goal_error_z_before"', '"goal_error_xy_after"',
        '"goal_error_z_after"', '"observation_provenance"',
        '"observation_provenance_after"',
    )
    assert all(field in source for field in required)


def test_update_zero_checkpoint_precedes_any_rollout_or_optimizer_use():
    source = inspect.getsource(pipeline._run_unlocked)
    save_zero = source.index('output / "checkpoint_update_000000"')
    rollout_loop = source.index("for update_index in range(")
    optimizer_attempt = source.index("_run_optimizer_attempt(")
    assert save_zero < rollout_loop < optimizer_attempt


def test_post_update_credit_audit_is_bound_before_metrics_and_checkpoint_publish():
    source = inspect.getsource(pipeline._run_unlocked)
    credit = source.index("evaluate_transition_credit_audit(")
    metrics = source.index('results / "metrics_update_', credit)
    checkpoint = source.index("\n            save_checkpoint(", credit)
    assert credit < metrics < checkpoint
    assert 'transition_record["dppo_credit"]' in source
    assert 'transition_record["gradient_diagnostics"]' in source
