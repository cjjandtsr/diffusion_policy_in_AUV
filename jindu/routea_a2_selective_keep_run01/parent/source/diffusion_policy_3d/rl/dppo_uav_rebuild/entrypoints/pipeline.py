"""Single canonical training loop for the E5 DDPM official-DPPO branch."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import re
import time
import uuid

import numpy as np
from omegaconf import OmegaConf
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.training_audit import (
    ParameterSnapshot,
    TrainingAudit,
    _scalar_stats,
    evaluate_post_update,
    evaluate_post_update_logical,
    evaluate_transition_credit_audit,
    run_audited_update,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import (
    restore_checkpoint,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.save import save_checkpoint
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import (
    CheckpointHeader,
    SCHEMA_VERSION,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import (
    OnPolicyRolloutBuffer,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.pair_schedule import (
    iter_pair_passes,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.plan_probe import (
    compare_probe_actions, sample_probe_action,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.update_transaction import (
    UpdateRejected,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_sampler import TaskSampler
from .build_system import build_system
from .identity import build_environment_provenance, build_runtime_identity
from .io_safety import RunLock


def _resolve(repo_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _bootstrap_allowed(terminated, truncated, *, finite_task):
    # `truncated` here is the environment's actual deadline, not the collector cut.
    return not (terminated or (finite_task and truncated))


def _write_json_once(path, payload):
    if path.exists():
        raise FileExistsError("refusing to overwrite artifact: {}".format(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_UPDATE_ARTIFACT_PATTERNS = (
    re.compile(r"^metrics_update_(\d{6})\.json$"),
    re.compile(r"^checkpoint_update_(\d{6})$"),
)


def _prepare_resume_output(output, start_update, report_dir=None):
    """Preserve stale post-checkpoint artifacts before deterministic replay."""

    output = Path(output).resolve()
    start_update = int(start_update)
    stale = []
    for artifact in output.iterdir():
        artifact_update = None
        for pattern in _UPDATE_ARTIFACT_PATTERNS:
            match = pattern.fullmatch(artifact.name)
            if match:
                artifact_update = int(match.group(1))
                break
        if artifact_update is not None and artifact_update >= start_update:
            stale.append(artifact)
    if not stale:
        return None

    restored_through = start_update - 1
    attempt = 1
    while True:
        quarantine = output / "resume_quarantine_after_{:06d}_{:03d}".format(
            restored_through, attempt
        )
        if not quarantine.exists():
            break
        attempt += 1
    quarantine.mkdir()
    moved = []
    for artifact in sorted(stale, key=lambda path: path.name):
        artifact.replace(quarantine / artifact.name)
        moved.append(artifact.name)
    _write_json_once(
        (quarantine / "resume_quarantine.json" if report_dir is None else
         Path(report_dir) / (quarantine.name + "_weights.json")),
        {
            "moved_artifacts": moved,
            "reason": "artifact update is newer than the restored checkpoint",
            "restored_through_update": restored_through,
            "resume_start_update": start_update,
        },
    )
    return quarantine


def _validate_run_config(config):
    """Reject run shapes that would make the formal-training budget ambiguous."""

    updates = int(config.run.updates)
    rollout_size = int(config.run.rollout_transitions)
    dppo_section = getattr(config, "dppo", None)
    v2_schedule = dppo_section is not None and "denoising_schedule" in dppo_section
    if v2_schedule:
        validate_v2_config(config)
    schedule_mode = (
        str(config.dppo.denoising_schedule) if v2_schedule
        else getattr(config.run, "pair_schedule_mode", None)
    )
    if schedule_mode is None:
        minibatch_size = int(getattr(config.run, "minibatch_size", 128))
        update_epochs = int(config.run.update_epochs)
    else:
        if "minibatch_size" in config.run:
            raise ValueError("minibatch_size is illegal with pair_schedule_mode")
        minibatch_size = 128
        update_epochs = int(config.dppo.pair_passes if v2_schedule else config.run.pair_passes)
    warmup_updates = int(getattr(config.run, "critic_warmup_updates", 0))
    checkpoint_every = int(getattr(config.run, "checkpoint_every_updates", 1))
    if updates <= 0 or rollout_size <= 0 or update_epochs <= 0:
        raise ValueError("updates, rollout_transitions, and update_epochs must be positive")
    if minibatch_size != 128:
        raise ValueError("official DPPO minibatch_size must remain 128")
    if rollout_size % minibatch_size != 0:
        raise ValueError("rollout_transitions must be divisible by minibatch_size")
    if warmup_updates < 0 or warmup_updates >= updates:
        raise ValueError("critic_warmup_updates must be non-negative and smaller than updates")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every_updates must be positive")
    actor_lr = float(config.optimizer.actor_lr)
    actor_lr_end = float(getattr(config.optimizer, "actor_lr_end", actor_lr))
    if actor_lr <= 0 or actor_lr_end <= 0:
        raise ValueError("actor learning-rate endpoints must be positive")


def _actor_lr_for_update(config, update_index):
    """Linear actor LR over joint actor-critic outer updates.

    The critic-only warmup update does not consume actor LR decay.  The first
    joint update receives actor_lr and the final outer update receives
    actor_lr_end exactly, which also makes resume deterministic from the outer
    update index without adding scheduler state to the checkpoint schema.
    """

    updates = int(config.run.updates)
    warmup_updates = int(getattr(config.run, "critic_warmup_updates", 0))
    update_index = int(update_index)
    if not 1 <= update_index <= updates:
        raise ValueError("update_index must be within the configured run")
    start = float(config.optimizer.actor_lr)
    end = float(getattr(config.optimizer, "actor_lr_end", start))
    first_joint = warmup_updates + 1
    if update_index <= first_joint:
        return start
    intervals = updates - first_joint
    if intervals <= 0:
        return end
    progress = (update_index - first_joint) / intervals
    return start + progress * (end - start)


def _should_save_checkpoint(config, update_index):
    update_index = int(update_index)
    return (
        update_index % int(getattr(config.run, "checkpoint_every_updates", 1)) == 0
        or update_index in tuple(int(i) for i in getattr(config.run, "checkpoint_updates", ()))
        or update_index == int(config.run.updates)
    )


def _execution_last_update(config, start_update, stop_after_update):
    """Bound one process at a committed checkpoint without changing run identity."""

    total_updates = int(config.run.updates)
    if stop_after_update is None:
        return total_updates
    if (isinstance(stop_after_update, bool)
            or int(stop_after_update) != stop_after_update):
        raise ValueError("stop_after_update must be an integer")
    stop = int(stop_after_update)
    if stop < int(start_update):
        raise ValueError("stop_after_update is before the resume start_update")
    if stop > total_updates:
        raise ValueError("stop_after_update exceeds configured updates")
    if not _should_save_checkpoint(config, stop):
        raise ValueError("stop_after_update must coincide with a checkpoint")
    return stop


def _run_optimizer_attempt(
    trainer,
    batch,
    *,
    update_actor,
    allow_recoverable_rejection=True,
):
    logical_microbatch_pairs = getattr(batch, "microbatch_pairs", None)
    try:
        evidence = run_audited_update(
            trainer,
            batch,
            update_actor=update_actor,
            evaluate_after=False,
            capture_parameter_delta=False,
            logical_microbatch_pairs=logical_microbatch_pairs,
        )
    except UpdateRejected as error:
        if not error.recoverable or not allow_recoverable_rejection:
            raise
        return {
            "status": "rejected",
            "reason": error.reason,
            "message": str(error),
            "pre_update": None,
            "post_update": None,
            "parameter_delta": None,
        }
    evidence["status"] = "committed"
    return evidence


def _update_outer_failure_streaks(counters, *, update_actor, actor_delta, committed):
    if committed:
        counters.consecutive_all_rejected_outer_updates = 0
    else:
        counters.consecutive_all_rejected_outer_updates += 1
    if update_actor and actor_delta == 0.0:
        counters.consecutive_zero_actor_delta_outer_updates += 1
    elif update_actor:
        counters.consecutive_zero_actor_delta_outer_updates = 0
    if counters.consecutive_all_rejected_outer_updates >= 5:
        return "five_consecutive_all_rejected_outer_updates"
    if counters.consecutive_zero_actor_delta_outer_updates >= 5:
        return "five_consecutive_zero_actor_delta_outer_updates"
    return None


def _run_unlocked(config, repo_root: Path, resume_from=None, stop_after_update=None):
    _validate_run_config(config)
    protocol = getattr(config, "task_protocol", None)
    finite_task = protocol is not None
    _execution_last_update(config, 1, stop_after_update)
    seed = int(config.run.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    system = build_system(config, repo_root)
    runtime_device = torch.device(system.bc_runtime.device)
    # Import only after the verified official top-level model package is loaded.
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import (
        RouteAEnvironmentFactory,
    )

    manifest = _resolve(repo_root, config.admission_manifest)
    worlds = _resolve(repo_root, config.worlds_train)
    output = _resolve(repo_root, config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = _resolve(repo_root, getattr(config, "result_dir", config.output_dir))
    results.mkdir(parents=True, exist_ok=True)
    environment_factory = RouteAEnvironmentFactory(
        manifest_path=manifest,
        worlds_train_dir=worlds,
        reward=system.reward,
        device=str(system.bc_runtime.device),
        base_seed=seed,
        max_episode_steps=int(config.environment.max_episode_steps),
        disturb_enable=bool(config.environment.disturb_enable),
        disturb_prob=float(config.environment.disturb_prob),
        disturb_pos_scale=float(config.environment.disturb_pos_scale),
        exec_noise_scale=float(config.environment.exec_noise_scale),
        mid_state_init_ratio=float(config.environment.mid_state_init_ratio),
        mid_state_pos_scale=float(config.environment.mid_state_pos_scale),
        max_reset_attempts=int(config.environment.max_reset_attempts),
        reward_config=OmegaConf.to_container(config.reward, resolve=True),
        terminate_at_space_limits=not finite_task,
        space_cost_budget=None if protocol is None else float(protocol.space_cost_budget),
    )
    task_sampler = TaskSampler(environment_factory.tasks, seed=seed + 17)
    runtime_identity = build_runtime_identity(config, repo_root, system, manifest)
    environment_provenance = build_environment_provenance(
        repo_root, reward_version=str(config.reward_version))
    r1_probe_path = results / "r1_plan_probe.json"
    r1_probe = None
    if str(config.reward_version) == "v2" and resume_from is not None and r1_probe_path.is_file():
        r1_probe = json.loads(r1_probe_path.read_text(encoding="utf-8"))
        if r1_probe["runtime_identity_hash"] != runtime_identity.content_hash:
            raise RuntimeError("R1 probe runtime identity mismatch on resume")

    start_update = 1
    resume_quarantine = None
    resume_metrics_quarantine = None
    if resume_from is not None:
        header = restore_checkpoint(
            resume_from,
            runtime_identity=runtime_identity,
            lora_identity=system.lora_bundle.identity,
            lora_bundle=system.lora_bundle,
            critic=system.critic,
            actor_optimizer=system.actor_optimizer,
            critic_optimizer=system.critic_optimizer,
            counters=system.trainer.counters,
            task_sampler=task_sampler,
            map_location=system.bc_runtime.device,
        )
        start_update = header.update_index + 1
        resume_quarantine = _prepare_resume_output(
            output, start_update, report_dir=results if results != output else None)
        if results != output:
            resume_metrics_quarantine = _prepare_resume_output(results, start_update)
        if str(config.reward_version) == "v2" and start_update > 1 and r1_probe is None:
            raise FileNotFoundError("R1 resume requires the frozen train-domain probe")
    elif 0 in tuple(int(value) for value in getattr(config.run, "checkpoint_updates", ())):
        initial_header = CheckpointHeader(
            schema_version=SCHEMA_VERSION,
            runtime_identity=asdict(runtime_identity),
            runtime_identity_hash=runtime_identity.content_hash,
            lora_identity_hash=system.lora_bundle.identity.content_hash,
            update_index=0,
        )
        save_checkpoint(
            output / "checkpoint_update_000000",
            training_config=OmegaConf.to_container(config, resolve=True) if finite_task else None,
            header=initial_header,
            lora_bundle=system.lora_bundle,
            critic=system.critic,
            actor_optimizer=system.actor_optimizer,
            critic_optimizer=system.critic_optimizer,
            counters=system.trainer.counters,
            task_sampler=task_sampler,
        )

    execution_last_update = _execution_last_update(
        config, start_update, stop_after_update)

    rollout_size = int(config.run.rollout_transitions)
    v2_schedule = "denoising_schedule" in config.dppo
    pair_schedule_mode = (
        str(config.dppo.denoising_schedule) if v2_schedule
        else getattr(config.run, "pair_schedule_mode", None)
    )
    minibatch_size = int(getattr(config.run, "minibatch_size", 128))
    pair_passes = int(
        config.dppo.pair_passes if v2_schedule else
        getattr(config.run, "pair_passes", getattr(config.run, "update_epochs", 1))
    )
    logical_batch_pairs = int(
        config.dppo.logical_batch_pairs if v2_schedule else
        getattr(config.run, "logical_batch_pairs", 128)
    )
    microbatch_pairs = int(
        config.dppo.microbatch_pairs if v2_schedule else
        getattr(config.run, "microbatch_pairs", 128)
    )
    warmup_updates = int(getattr(config.run, "critic_warmup_updates", 0))
    total_updates = int(config.run.updates)
    print(
        json.dumps(
            {
                "event": "dppo_training_start",
                "initialization": "checkpoint_resume" if resume_from else "fresh_bc",
                "bc_checkpoint": str(_resolve(repo_root, config.checkpoint)),
                "resume_from": str(resume_from) if resume_from else None,
                "output_dir": str(output),
                "result_dir": str(results),
                "start_update": start_update,
                "total_updates": total_updates,
                "execution_last_update": execution_last_update,
                "planned_transitions": total_updates * rollout_size,
                "runtime_identity_hash": runtime_identity.content_hash,
                "initialization_parity": system.initialization_parity,
                "environment_provenance": environment_provenance,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    for update_index in range(start_update, execution_last_update + 1):
        update_started = time.perf_counter()
        if runtime_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime_device)
        task_role_episode_start_counts = Counter()
        audit = TrainingAudit(term_names=getattr(system.reward, "term_names", None),
                              terminate_at_space_limits=(environment_factory.terminate_at_space_limits
                                                         if str(config.reward_version) == "v2" else None))
        buffer = OnPolicyRolloutBuffer(
            rollout_size,
            gamma=float(config.dppo.gamma_environment),
            gae_lambda=float(config.dppo.gae_lambda),
            outer_update_id=update_index,
            history_scalar=str(config.reward_version) == "v2" and not finite_task,
            episode_context=finite_task,
        )
        adapter = None
        observation = None
        observation_context = None
        encoded_observation = None
        task = None
        reset_seed = None
        while len(buffer) < rollout_size:
            if adapter is None:
                task = task_sampler.sample()
                task_role_episode_start_counts[task.role] += 1
                adapter = environment_factory.make(task)
                reset_seed = task.seed + system.trainer.counters.rollout_segments * 1000003
                observation, observation_context = adapter.reset(seed=reset_seed)
            if encoded_observation is None:
                encoded_observation = system.observation_adapter.encode(
                    observation,
                    observation_context["pointcloud_meta_history"],
                    episode_identity_hash=observation_context["episode_identity_hash"],
                    decision_index=observation_context["decision_index"],
                )
            system.observation_adapter.verify(encoded_observation)
            condition = encoded_observation.condition
            if str(config.reward_version) == "v2" and r1_probe is None:
                initial_action = sample_probe_action(
                    system.rollout_sampler, condition, seed=seed + 981731,
                    device=runtime_device)
                r1_probe = {
                    "runtime_identity_hash": runtime_identity.content_hash,
                    "probe_seed": seed + 981731,
                    "condition": {key: value.detach().cpu().tolist()
                                  for key, value in condition.items()},
                    "initial_action": initial_action.tolist(),
                    "source": "first admitted train-domain observation at update 1",
                }
                _write_json_once(r1_probe_path, r1_probe)
            if str(config.reward_version) == "v2":
                from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.critic_state import with_history_scalar, with_episode_context
                critic_condition = (with_episode_context(condition, adapter.environment) if finite_task
                                    else with_history_scalar(condition, adapter.environment._reward_history))
            else:
                critic_condition = condition
            position_before = np.asarray(adapter.environment._drone_pos, dtype=np.float64).copy()
            goal_xyz = np.asarray(adapter.environment.goal_w, dtype=np.float64).copy()
            goal_distance_before = float(np.linalg.norm(position_before - goal_xyz))
            with torch.no_grad():
                value = float(system.critic(critic_condition).reshape(-1)[0].item())
                system.observation_adapter.verify(encoded_observation)
                rollout = system.rollout_sampler.sample(condition, deterministic=False)
            next_observation, reward, terminated, truncated, info = adapter.step(
                rollout.action
            )
            if info.get("reward_metrics", {}).get("training_usable") is False:
                raise FloatingPointError("invalid_state before buffer/critic update: {}".format(
                    info.get("reward_metrics")))
            collection_boundary = len(buffer) + 1 == rollout_size
            effective_truncated = bool(truncated or (collection_boundary and not terminated))
            bootstrap_allowed = _bootstrap_allowed(terminated, truncated, finite_task=finite_task)
            next_encoded_observation = system.observation_adapter.encode(
                next_observation,
                info["pointcloud_meta_history"],
                episode_identity_hash=info["episode_identity_hash"],
                decision_index=info["decision_index"],
            )
            system.observation_adapter.verify(next_encoded_observation)
            next_condition = next_encoded_observation.condition
            if str(config.reward_version) == "v2":
                next_critic_condition = (with_episode_context(next_condition, adapter.environment) if finite_task
                                         else with_history_scalar(next_condition, adapter.environment._reward_history))
            else:
                next_critic_condition = next_condition
            with torch.no_grad():
                next_value = (
                    0.0
                    if not bootstrap_allowed
                    else float(system.critic(next_critic_condition).reshape(-1)[0].item())
                )
            audit.record_transition(
                task_id=task.task_id,
                task_role=task.role,
                reward=reward,
                value=value,
                next_value=next_value,
                terminated=terminated,
                truncated=truncated,
                collection_boundary=collection_boundary,
                bootstrap_allowed=bootstrap_allowed,
                critic_context=(critic_condition["episode_context"][0].detach().cpu().tolist()
                                if finite_task else None),
                next_critic_context=(next_critic_condition["episode_context"][0].detach().cpu().tolist()
                                     if finite_task else None),
                info=info,
                action_chunk=rollout.action.rollout_executed[0],
                audit_context={
                    "run_seed": seed,
                    "update_index": update_index,
                    "policy_state_update_index": update_index - 1,
                    "transition_id": "u{}:t{}".format(update_index, len(buffer)),
                    "episode_id": encoded_observation.provenance.episode_identity_hash,
                    "decision_index": encoded_observation.provenance.decision_index,
                    "task_identity_hash": task.identity_hash,
                    "world_id": task.world_id,
                    "task_seed": task.seed,
                    "reset_seed": reset_seed,
                    "admission_manifest_hash": runtime_identity.task_manifest_hash,
                    "runtime_identity_hash": runtime_identity.content_hash,
                    "source_set_hash": runtime_identity.source_set_hash,
                    "config_hash": runtime_identity.config_hash,
                    "checkpoint_identity_hash": runtime_identity.bc_identity_hash,
                    "goal_xyz": goal_xyz.tolist(),
                    "position_before_xyz": position_before.tolist(),
                    "position_after_xyz": np.asarray(adapter.environment._drone_pos, dtype=np.float64).tolist(),
                    "goal_distance_before_3d": goal_distance_before,
                    "goal_distance_after_3d": float(np.linalg.norm(np.asarray(adapter.environment._drone_pos, dtype=np.float64) - goal_xyz)),
                    "goal_error_xy_before": float(np.linalg.norm(position_before[:2] - goal_xyz[:2])),
                    "goal_error_z_before": float(abs(position_before[2] - goal_xyz[2])),
                    "goal_error_xy_after": float(np.linalg.norm(np.asarray(adapter.environment._drone_pos, dtype=np.float64)[:2] - goal_xyz[:2])),
                    "goal_error_z_after": float(abs(np.asarray(adapter.environment._drone_pos, dtype=np.float64)[2] - goal_xyz[2])),
                    "observation_provenance": asdict(encoded_observation.provenance),
                    "observation_provenance_after": asdict(next_encoded_observation.provenance),
                },
            )
            system.observation_adapter.verify(encoded_observation)
            buffer.add(
                condition={key: value_[0] for key, value_ in critic_condition.items()},
                chain=rollout.normalized_chain[0],
                old_logprobs=rollout.old_logprobs[0],
                reward=reward,
                value=value,
                next_value=next_value,
                terminated=terminated,
                truncated=effective_truncated,
                bootstrap_allowed=bootstrap_allowed,
                transition_id="u{}:t{}".format(update_index, len(buffer)),
                role=task.role,
                outcome=info["transition_facts"].done_reason,
                provenance_hash=encoded_observation.provenance.policy_input_hash,
            )
            system.trainer.counters.env_steps += 1
            observation = next_observation
            observation_context = info
            encoded_observation = next_encoded_observation
            if terminated or truncated or collection_boundary:
                system.trainer.counters.rollout_segments += 1
                if terminated or truncated:
                    system.trainer.counters.completed_episodes += 1
                adapter = None
                observation = None
                observation_context = None
                encoded_observation = None
                task = None

        buffer.finalize()
        rollout_audit = audit.finalize(buffer.audit_tensors())
        rollout_finished = time.perf_counter()
        actor_lr = _actor_lr_for_update(config, update_index)
        for group in system.actor_optimizer.param_groups:
            group["lr"] = actor_lr
        update_actor = update_index > warmup_updates
        update_metrics = []
        rejection_reason_counts = Counter()
        outer_snapshot = ParameterSnapshot.capture(
            actor=system.trainer.actor_parameters,
            critic=system.trainer.critic_parameters,
        )
        last_committed_batch = None
        schedule_audits = []
        pair_evaluations = 0
        microbatch_evaluations = 0
        if pair_schedule_mode is None:
            expected_optimizer_steps = pair_passes * rollout_size // minibatch_size
            batch_iterator = (
                batch
                for epoch in range(pair_passes)
                for batch in buffer.batches(
                    minibatch_size,
                    generator=torch.Generator().manual_seed(
                        seed + update_index * 1009 + epoch
                    ),
                    device=system.bc_runtime.device,
                )
            )
        else:
            total_pairs_per_pass = (
                rollout_size if pair_schedule_mode == "sampled_k" else rollout_size * 10
            )
            expected_optimizer_steps = (
                pair_passes * total_pairs_per_pass // logical_batch_pairs
            )
            batch_iterator = iter_pair_passes(
                buffer,
                mode=str(pair_schedule_mode),
                pair_passes=pair_passes,
                logical_batch_pairs=logical_batch_pairs,
                microbatch_pairs=microbatch_pairs,
                generator=torch.Generator().manual_seed(seed + update_index * 1009),
                device=system.bc_runtime.device,
            )
        for batch in batch_iterator:
            if hasattr(batch, "pair_count"):
                pair_evaluations += batch.pair_count
                microbatch_evaluations += batch.microbatch_count
                if not schedule_audits or schedule_audits[-1]["pass_id"] != batch.pass_id:
                    schedule_audits.append(batch.schedule_audit)
            else:
                pair_evaluations += int(batch.chains_prev.shape[0])
                microbatch_evaluations += 1
            evidence = _run_optimizer_attempt(
                system.trainer,
                batch,
                update_actor=update_actor,
                allow_recoverable_rejection=update_actor,
            )
            update_metrics.append(evidence)
            if evidence["status"] == "rejected":
                rejection_reason_counts[evidence["reason"]] += 1
            else:
                last_committed_batch = batch
        if len(update_metrics) != expected_optimizer_steps:
            raise RuntimeError(
                "optimizer step count mismatch: expected {}, got {}".format(
                    expected_optimizer_steps, len(update_metrics)
                )
            )
        committed_metrics = [
            item for item in update_metrics if item["status"] == "committed"
        ]
        registered_commits = getattr(config.run, "expected_optimizer_commits_per_outer_update", None)
        if registered_commits is not None and len(committed_metrics) != int(registered_commits):
            raise RuntimeError("committed optimizer count differs from the pre-registered v2 contract")
        optimizer_metric_summary = {
            name: _scalar_stats(
                [item["pre_update"][name] for item in committed_metrics]
            )
            for name in (
                "policy_loss",
                "value_loss",
                "entropy_loss",
                "total_loss",
                "approx_kl",
                "ratio_mean",
                "clip_fraction",
                "actor_grad_norm",
                "critic_grad_norm",
            )
        }
        if last_committed_batch is not None:
            if hasattr(last_committed_batch, "microbatch_pairs"):
                post_update = evaluate_post_update_logical(
                    system.trainer, last_committed_batch,
                    microbatch_pairs=last_committed_batch.microbatch_pairs,
                    update_actor=update_actor,
                )
            else:
                post_update = evaluate_post_update(
                    system.trainer, last_committed_batch, update_actor=update_actor,
                )
            committed_metrics[-1]["post_update"] = post_update
        outer_parameter_delta = outer_snapshot.measure(
            actor=system.trainer.actor_parameters,
            critic=system.trainer.critic_parameters,
        )
        hard_stop_reason = _update_outer_failure_streaks(
            system.trainer.counters,
            update_actor=update_actor,
            actor_delta=outer_parameter_delta["actor"]["l2_norm"],
            committed=bool(committed_metrics),
        )
        optimizer_finished = time.perf_counter()
        credit_audit_started = time.perf_counter()
        transition_credit_records = evaluate_transition_credit_audit(
            system.trainer,
            buffer,
            microbatch_pairs=min(128, int(logical_batch_pairs or minibatch_size)),
        )
        transition_credit_by_id = {
            record["transition_id"]: dict(record)
            for record in transition_credit_records
        }
        rollout_transition_ids = {
            record["transition_id"]
            for record in rollout_audit["transition_records"]
        }
        if set(transition_credit_by_id) != rollout_transition_ids:
            raise RuntimeError("transition credit audit identity coverage mismatch")
        gradient_diagnostics = {
            "scope": "outer_update",
            "actor_grad_norm": optimizer_metric_summary["actor_grad_norm"],
            "critic_grad_norm": optimizer_metric_summary["critic_grad_norm"],
            "outer_parameter_delta": outer_parameter_delta,
            "committed_optimizer_steps": len(committed_metrics),
            "rejected_optimizer_steps": len(update_metrics) - len(committed_metrics),
        }
        for transition_record in rollout_audit["transition_records"]:
            transition_record["dppo_credit"] = transition_credit_by_id[
                transition_record["transition_id"]
            ]
            transition_record["gradient_diagnostics"] = gradient_diagnostics
        credit_audit_seconds = time.perf_counter() - credit_audit_started

        checkpoint_path = None
        checkpoint_header = None
        if _should_save_checkpoint(config, update_index):
            checkpoint_header = CheckpointHeader(
                schema_version=SCHEMA_VERSION,
                runtime_identity=asdict(runtime_identity),
                runtime_identity_hash=runtime_identity.content_hash,
                lora_identity_hash=system.lora_bundle.identity.content_hash,
                update_index=update_index,
            )
            checkpoint_path = output / "checkpoint_update_{:06d}".format(update_index)
        r1_probe_metrics = None
        if checkpoint_path is not None and str(config.reward_version) == "v2":
            probe_condition = {
                key: torch.tensor(value, device=runtime_device, dtype=torch.float32)
                for key, value in r1_probe["condition"].items()
            }
            current_action = sample_probe_action(
                system.rollout_sampler, probe_condition,
                seed=r1_probe["probe_seed"], device=runtime_device)
            r1_probe_metrics = compare_probe_actions(
                r1_probe["initial_action"], current_action)
        rollout_seconds = rollout_finished - update_started
        optimizer_seconds = optimizer_finished - rollout_finished
        total_seconds = time.perf_counter() - update_started
        peak_cuda_memory_bytes = (
            int(torch.cuda.max_memory_allocated(runtime_device))
            if runtime_device.type == "cuda"
            else 0
        )
        _write_json_once(
            results / "metrics_update_{:06d}.json".format(update_index),
            {
                "audit_schema_version": 5,
                "environment_provenance": environment_provenance,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "r1_plan_probe": r1_probe_metrics,
                "checkpoint_expected_path": (
                    str(checkpoint_path) if checkpoint_path else None
                ),
                "checkpoint_publish_state_at_metrics_write": (
                    "pending" if checkpoint_path else "not_scheduled"
                ),
                "counters": vars(system.trainer.counters),
                "formal_training": {
                    "actor_learning_rate": actor_lr,
                    "min_sampling_std": float(config.dppo.min_sampling_std),
                    "min_logprob_std": float(config.dppo.min_logprob_std),
                    "mode": "joint" if update_actor else "critic_only_warmup",
                    "optimizer_steps": len(update_metrics),
                    "pair_schedule_mode": str(pair_schedule_mode) if pair_schedule_mode else "legacy_sampled_k",
                    "pair_evaluations": pair_evaluations,
                    "microbatch_evaluations": microbatch_evaluations,
                    "logical_commits_attempted": len(update_metrics),
                    "pair_schedule_audits": schedule_audits,
                    "committed_optimizer_steps": len(committed_metrics),
                    "rejected_optimizer_steps": (
                        len(update_metrics) - len(committed_metrics)
                    ),
                    "rejection_reason_counts": dict(
                        sorted(rejection_reason_counts.items())
                    ),
                    "hard_stop_reason": hard_stop_reason,
                    "outer_parameter_delta": outer_parameter_delta,
                    "optimizer_metric_summary": optimizer_metric_summary,
                    "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
                    "rollout_seconds": rollout_seconds,
                    "rollout_transitions_per_second": rollout_size / rollout_seconds,
                    "resume_quarantine": (
                        str(resume_quarantine) if resume_quarantine else None
                    ),
                    "resume_metrics_quarantine": (
                        str(resume_metrics_quarantine) if resume_metrics_quarantine else None
                    ),
                    "task_role_episode_start_counts": dict(
                        sorted(task_role_episode_start_counts.items())
                    ),
                    "task_role_transition_counts": rollout_audit["episodes"][
                        "role_transition_counts"
                    ],
                    "total_seconds": total_seconds,
                    "update_seconds": optimizer_seconds,
                    "transition_credit_audit_seconds": credit_audit_seconds,
                    "transition_credit_audit_count": len(
                        transition_credit_records
                    ),
                },
                "rollout_audit": rollout_audit,
                "runtime_identity_hash": runtime_identity.content_hash,
                "updates": update_metrics,
            },
        )
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                training_config=OmegaConf.to_container(config, resolve=True) if finite_task else None,
                header=checkpoint_header,
                lora_bundle=system.lora_bundle,
                critic=system.critic,
                actor_optimizer=system.actor_optimizer,
                critic_optimizer=system.critic_optimizer,
                counters=system.trainer.counters,
                rollout_buffer=buffer,
                task_sampler=task_sampler,
            )
        latest_update = committed_metrics[-1] if committed_metrics else None
        latest_post_update = latest_update["post_update"] if latest_update else None
        print(
            json.dumps(
                {
                    "event": "dppo_outer_update_complete",
                    "update_index": update_index,
                    "mode": "joint" if update_actor else "critic_only_warmup",
                    "env_steps": system.trainer.counters.env_steps,
                    "actor_learning_rate": actor_lr,
                    "optimizer_steps": len(update_metrics),
                    "committed_optimizer_steps": len(committed_metrics),
                    "rejected_optimizer_steps": (
                        len(update_metrics) - len(committed_metrics)
                    ),
                    "rejection_reason_counts": dict(
                        sorted(rejection_reason_counts.items())
                    ),
                    "hard_stop_reason": hard_stop_reason,
                    "rollout_transitions_per_second": rollout_size / rollout_seconds,
                    "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
                    "reward_mean": rollout_audit["reward"]["total"]["mean"],
                    "completed_episodes": rollout_audit["episodes"]["completed"],
                    "physical_done_reason_counts": rollout_audit["episodes"][
                        "physical_done_reason_counts"
                    ],
                    "actor_delta_l2": outer_parameter_delta["actor"]["l2_norm"],
                    "critic_delta_l2": outer_parameter_delta["critic"]["l2_norm"],
                    "post_update_approx_kl": (
                        latest_post_update["approx_kl"]
                        if latest_post_update
                        else None
                    ),
                    "post_update_ratio_mean": (
                        latest_post_update["ratio_mean"]
                        if latest_post_update
                        else None
                    ),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        if hard_stop_reason is not None:
            raise RuntimeError("formal training hard stop: {}".format(hard_stop_reason))
    if execution_last_update < total_updates:
        print(json.dumps({
            "event": "dppo_training_paused_at_checkpoint",
            "completed_through_update": execution_last_update,
            "configured_total_updates": total_updates,
            "checkpoint": str(output / "checkpoint_update_{:06d}".format(
                execution_last_update)),
            "runtime_identity_hash": runtime_identity.content_hash,
        }, sort_keys=True, allow_nan=False), flush=True)
    return system.trainer.counters


def run(config, repo_root: Path, resume_from=None, stop_after_update=None):
    """Hold one exclusive writer lock for the complete run transaction."""

    results = _resolve(Path(repo_root), getattr(config, "result_dir", config.output_dir))
    owner_token = "pid{}-{}".format(os.getpid(), uuid.uuid4().hex)
    lock = RunLock.acquire(results, owner_token)
    try:
        return _run_unlocked(config, repo_root, resume_from=resume_from,
                             stop_after_update=stop_after_update)
    finally:
        lock.release()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[4]
    config = OmegaConf.load(args.config)
    run(config, repo_root, resume_from=args.resume_from,
        stop_after_update=args.stop_after_update)
    return 0
