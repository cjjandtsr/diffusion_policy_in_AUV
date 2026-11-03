"""Strict Route A v1.7 pilot/formal schedule schema."""

from __future__ import annotations

from dataclasses import dataclass
import re
import math


@dataclass(frozen=True)
class ScheduleContract:
    schedule: str
    pair_passes: int
    logical_batch_pairs: int
    microbatch_pairs: int
    expected_optimizer_commits_per_outer_update: int


_CONTRACTS = {
    "S0": ScheduleContract("sampled_k", 5, 128, 128, 40),
    "S2": ScheduleContract("full_joint_accum_batch1280", 5, 1280, 128, 40),
}

_TOP_LEVEL = {"checkpoint", "vendor_root", "admission_manifest", "worlds_train", "output_dir", "result_dir", "device", "run", "lora", "critic", "optimizer", "dppo", "gates", "environment", "reward", "reward_version", "reward_density", "reward_calibration", "task_protocol", "a2", "xk"}
_RUN_KEYS = {"phase", "seed", "updates", "rollout_transitions", "schedule_label", "checkpoint_updates", "expected_optimizer_commits_per_outer_update", "checkpoint_every_updates"}
_DPPO_KEYS = {"gamma_environment", "gae_lambda", "gamma_denoising", "clip_ploss_coef", "min_sampling_std", "min_logprob_std", "value_loss_coefficient", "entropy_loss_coefficient", "denoising_schedule", "pair_passes", "logical_batch_pairs", "microbatch_pairs", "ppo_inert_mask", "drift_monitor", "drift_gate", "inert_anchor", "self_imitation", "min_std_vector", "denoised_clip_value"}
_ENVIRONMENT_KEYS = {"max_episode_steps", "disturb_enable", "disturb_prob", "disturb_pos_scale", "exec_noise_scale", "mid_state_init_ratio", "mid_state_pos_scale", "max_reset_attempts", "goal_tolerance_3d", "reject_all_no_ray_hit", "max_no_ray_hit_fraction", "pointcloud_calibration_sha256"}
_OPTIMIZER_KEYS = {"actor_lr", "actor_lr_end", "critic_lr", "weight_decay"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_R1_REWARD = {
    "H_e": 4, "max_episode_steps": 60, "gamma_pbrs": .99,
    "r_reach": 100., "r_failure": 50., "newbest_budget": 10.,
    "risk_weight": .5, "time_cost": .05, "d_safe": .45,
    "collision_margin_xy": .1, "ground_safe": 1., "ground_contact": .2,
}


def validate_v2_config(config):
    _validate_optional_f5(config)
    _validate_optional_xk(config)
    if 'a2' in config:
        from .a2_support import validate_a2_config
        validate_a2_config(config)
    if "minibatch_size" in config.run:
        raise ValueError("minibatch_size is forbidden by the v2 pair schedule schema")
    unknown_top = set(config.keys()) - (_TOP_LEVEL | {'f5', 'task_bank'})
    unknown_run = set(config.run.keys()) - _RUN_KEYS
    unknown_dppo = set(config.dppo.keys()) - _DPPO_KEYS
    unknown_environment = set(config.environment.keys()) - _ENVIRONMENT_KEYS
    unknown_optimizer = set(config.optimizer.keys()) - _OPTIMIZER_KEYS
    if unknown_top or unknown_run or unknown_dppo or unknown_environment or unknown_optimizer:
        raise ValueError("unknown v2 config fields: top={} run={} dppo={} environment={} optimizer={}".format(sorted(unknown_top), sorted(unknown_run), sorted(unknown_dppo), sorted(unknown_environment), sorted(unknown_optimizer)))
    run = config.run
    dppo = config.dppo
    reward_version = str(config.reward_version)
    protocol = config.get("task_protocol")
    if protocol is not None:
        if (set(protocol) != {"name", "space_cost_budget"} or
                protocol.name != "open_space_finite_v1" or reward_version not in ("v2", "v3")):
            raise ValueError("unknown common task protocol")
        budget = float(protocol.space_cost_budget)
        if not math.isfinite(budget) or budget < 0.:
            raise ValueError("space_cost_budget must be finite and non-negative; zero is ablation only")
        if int(config.environment.max_episode_steps) != 60:
            raise ValueError("common finite task requires 60 decisions / 72 seconds")
        if str(run.phase) != "formal" or int(run.updates) != 256:
            raise ValueError("common protocol comparison requires the planned 256 updates")
    if reward_version == "v17":
        if int(config.reward.H_e) != 4:
            raise ValueError("v2 A0 training requires Reward v1.7 with H_e=4")
    elif reward_version in ("v2", "v3"):
        if set(config.reward.keys()) != set(_R1_REWARD) or any(
            float(config.reward[key]) != expected for key, expected in _R1_REWARD.items()
        ):
            raise ValueError("R1 reward coefficients must match frozen v1.4 design")
        if (float(dppo.gamma_environment) != .99 or
                float(dppo.gamma_environment) != float(config.reward.gamma_pbrs)):
            raise ValueError("R1 gamma_environment must equal gamma_pbrs=0.99")
        if "reward_density" in config or "reward_calibration" in config:
            raise ValueError("R1 must not load v1.7 reward calibration artifacts")
    else:
        raise ValueError("v2 training requires explicit A0 v17 or R1 v2 reward")
    if float(config.optimizer.actor_lr) != 2.0e-5 or float(config.optimizer.actor_lr_end) != 1.0e-5:
        raise ValueError("v2 actor learning-rate identity mismatch")
    if float(config.dppo.clip_ploss_coef) not in ((.1,.05) if 'xk' in config else (.1,)):
        raise ValueError("v2 clip_ploss_coef identity mismatch")
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import canonical_environment_sampling_spec
    canonical_environment_sampling_spec(config.environment)
    if not isinstance(config.environment.reject_all_no_ray_hit, bool):
        raise TypeError("reject_all_no_ray_hit must be boolean")
    threshold = config.environment.max_no_ray_hit_fraction
    if threshold is None:
        raise ValueError("max_no_ray_hit_fraction requires frozen BC calibration")
    if not 0.0 <= float(threshold) < 1.0:
        raise ValueError("max_no_ray_hit_fraction must be in [0,1)")
    if "checkpoint" in config:
        calibration_hash = str(config.environment.pointcloud_calibration_sha256)
        if not _SHA256.fullmatch(calibration_hash):
            raise ValueError("resolved v2 config requires pointcloud_calibration_sha256")
        if reward_version == "v17":
            for name in ("reward_density", "reward_calibration"):
                entry = getattr(config, name, None)
                if entry is None or not _SHA256.fullmatch(str(entry.sha256)):
                    raise ValueError("resolved v2 config requires frozen {} identity".format(name))
    label = str(run.schedule_label)
    if protocol is not None:
        logical = int(dppo.logical_batch_pairs)
        if label != "S2_common" or logical not in (1280, 2560):
            raise ValueError("common protocol requires S2_common with 1280 or 2560 pairs")
        contract = ScheduleContract("full_joint_accum", 5, logical, 128, 51200 // logical)
    elif label not in _CONTRACTS:
        raise ValueError("schedule_label must be S0 or S2")
    else:
        contract = _CONTRACTS[label]
    if protocol is None and reward_version == "v2" and label != "S2":
        raise ValueError("R1 is frozen to the matched A0 S2 schedule")
    observed = ScheduleContract(
        str(dppo.denoising_schedule), int(dppo.pair_passes),
        int(dppo.logical_batch_pairs), int(dppo.microbatch_pairs),
        int(run.expected_optimizer_commits_per_outer_update),
    )
    if observed != contract:
        raise ValueError("{} schedule contract mismatch: {} != {}".format(label, observed, contract))
    if str(run.phase) == "pilot":
        if int(run.seed) not in (202609101, 202609102, 202609103):
            raise ValueError("pilot seed is not pre-registered")
        if int(run.updates) != 32 or int(run.rollout_transitions) != 1024:
            raise ValueError("pilot requires 32 updates and 1024 transitions")
        if tuple(int(value) for value in run.checkpoint_updates) != (0, 8, 16, 24, 32):
            raise ValueError("pilot checkpoint updates must be 0,8,16,24,32")
    elif str(run.phase) == "formal":
        if reward_version == "v2" and 'f5' not in config and int(run.seed) not in ((202610261,) if 'xk' in config else (202609201, 202609202, 202609203)):
            raise ValueError("R1 formal seed must match one of the three A0 seeds")
        updates = int(run.updates)
        if updates < 256 or updates % 8 or int(run.rollout_transitions) != 1024:
            raise ValueError("formal requires at least 256 updates at interval 8 and 1024 transitions")
        checkpoints = (0, 1) + tuple(range(8, updates + 1, 8)) if protocol is not None else tuple(range(0, updates + 1, 8))
        if 'a2' in config:
            checkpoints = tuple(sorted(set(checkpoints) | {129}))
        if tuple(int(value) for value in run.checkpoint_updates) != checkpoints:
            raise ValueError("formal checkpoint updates must be 0..updates at interval 8 (plus 1 for common protocol)")
    else:
        raise ValueError("run.phase must be pilot or formal")
    return contract


def _validate_optional_f5(config):
    bank = config.get('task_bank')
    if 'f5' not in config:
        if str(config.get('reward_version')) == 'v3' or (bank is not None and dict(bank) != {'enabled': False}):
            raise ValueError('task bank and reward v3 require a registered F5 experiment')
        return
    spec = config.f5
    registered = {'P8': (202610311, True), 'P9': (202610311, False),
                  'P10': (202610312, True), 'P11': (202610312, False)}
    if set(spec) != {'run_id', 'keep_coefficient'} or spec.run_id not in registered:
        raise ValueError('invalid F5 run registration')
    seed, treatment = registered[spec.run_id]
    if (int(config.run.seed) != seed or float(spec.keep_coefficient) != .01061006
            or 'xk' in config or 'a2' not in config or config.a2.arm != 'T'
            or int(config.a2.keep_seed) != 2026102301 or 'task_protocol' not in config
            or str(config.reward_version) != ('v3' if treatment else 'v2')):
        raise ValueError('F5 registered seed, keep or reward differs')
    extra = set(config.dppo) & {'ppo_inert_mask', 'self_imitation', 'inert_anchor',
                               'min_std_vector', 'denoised_clip_value', 'drift_monitor', 'drift_gate'}
    if extra:
        raise ValueError('F5 does not include P7 sampling or auxiliary changes')
    if not treatment:
        if bank is not None and dict(bank) != {'enabled': False}:
            raise ValueError('F3 control must use the original sampler')
        return
    mixture = {'original': .15, 'L0': .10, 'L1': .15,
               'L23_random': .30, 'L23_near': .20, 'L4': .10}
    if (bank is None or set(bank) != {'enabled', 'path', 'sha256', 'seed', 'mixture'}
            or bank.enabled is not True or not str(bank.path)
            or not _SHA256.fullmatch(str(bank.sha256))
            or int(bank.seed) != seed + 17 or dict(bank.mixture) != mixture):
        raise ValueError('F5 task bank configuration differs from the fixed mixture')


def _validate_optional_xk(config):
    d=config.dppo
    if 'ppo_inert_mask' in d:
        from ..engine.official_loss_components import InertMask
        if set(d.ppo_inert_mask) != {'enabled','channels','scope'}:
            raise ValueError('invalid inert mask fields')
        InertMask(**dict(d.ppo_inert_mask))
    if 'min_std_vector' in d:
        values=list(d.min_std_vector)
        if len(values)!=11 or any(not math.isfinite(float(v)) or float(v)<=0 for v in values):
            raise ValueError('min_std_vector requires eleven positive finite values')
    if 'denoised_clip_value' in d and (not math.isfinite(float(d.denoised_clip_value)) or float(d.denoised_clip_value)<=0):
        raise ValueError('invalid denoised clip override')
    if 'drift_gate' in d and (set(d.drift_gate)!={'enabled'} or d.drift_gate.enabled is not False):
        raise ValueError('hard drift gate is not implemented or approved')
    if 'drift_monitor' in d:
        monitor=d.drift_monitor
        if (set(monitor)!={'enabled','bank_path','bank_sha256','every_commits'}
                or not isinstance(monitor.enabled,bool) or int(monitor.every_commits)!=5
                or not _SHA256.fullmatch(str(monitor.bank_sha256))):
            raise ValueError('invalid drift monitor config')
    if 'inert_anchor' in d:
        anchor=d.inert_anchor
        if (set(anchor)!={'enabled','coefficient','target_ratio','batch_size','seed'}
                or not isinstance(anchor.enabled,bool) or float(anchor.target_ratio)!=.05
                or int(anchor.batch_size)!=128 or not math.isfinite(float(anchor.coefficient))
                or float(anchor.coefficient)<0):
            raise ValueError('invalid inert anchor config')
    if 'self_imitation' in d:
        imitation=d.self_imitation
        if (set(imitation)!={'enabled','window_updates','capacity','target_ratio','batch_size','seed'}
                or not isinstance(imitation.enabled,bool) or int(imitation.window_updates)!=8
                or int(imitation.capacity)!=2048 or float(imitation.target_ratio)!=.1
                or int(imitation.batch_size)!=32):
            raise ValueError('invalid rolling self imitation config')
    if 'xk' in config:
        x=config.xk
        if (set(x)-{'arm','actor_lr_scale','c1_gradient_output'} or
                not {'arm','actor_lr_scale'}<=set(x) or x.arm not in ('X','K') or
                not math.isfinite(float(x.actor_lr_scale)) or float(x.actor_lr_scale)<=0):
            raise ValueError('invalid X/K config')
        if ('a2' not in config or config.a2.arm!='C' or
                not d.get('drift_monitor',{}).get('enabled',False)):
            raise ValueError('X/K requires explicit parent fork and fixed drift monitoring')
        enabled=d.get('self_imitation',{}).get('enabled',False)
        masked=d.get('ppo_inert_mask',{}).get('enabled',False)
        if bool(enabled)!=(x.arm=='X') or bool(masked)!=(x.arm=='X'):
            raise ValueError('X requires T1 and T5; K disables both')
        if d.get('inert_anchor',{}).get('enabled',False):
            raise ValueError('T4 remains disabled for this X/K run')


__all__ = ("ScheduleContract", "validate_v2_config")
