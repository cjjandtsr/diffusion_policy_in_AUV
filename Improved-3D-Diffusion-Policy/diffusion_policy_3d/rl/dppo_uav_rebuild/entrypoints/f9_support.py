"""Online teacher sidecar for the canonical pipeline; no guided PPO transitions."""
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import importlib.util
import inspect
import sys

import numpy as np
import torch

from ..engine.f9_online import OnlineKeep, capture_rng, rng_scope, paired_sample
from ..engine.contracts import ActionBundle
from ..environments.task_sampler import TaskSampler


def validate_f9(config):
    f = config.f9
    assert 'f8' in config and not config.f8.bank_path
    assert int(config.run.seed) in (202611081, 202611082, 202611213)
    assert float(f.d_safe) in (.6, .9, 1.2)
    assert (float(f.eta), int(f.M)) in ((.2, 2), (.4, 4))
    assert int(f.capacity) == 4096 and int(f.batch_size) == 32
    assert float(f.target_ratio) == 1. and float(f.effective_threshold) == .05
    assert float(f.coefficient_override) >= 0
    assert f.stage in ('A', 'B') and int(f.attempt) in (1, 2)
    assert str(f.get('teacher_task_sampler', 'default')) in ('default', 'blocked')
    assert bool(f.restart_reference) == (int(f.attempt) == 2)
    assert (float(f.coefficient_override) > 0) == (int(f.attempt) == 2)


class F9Session:
    def __init__(self, system, config, factory, results):
        self.system, self.config = system, config
        self.directory = Path(results) / 'f9'
        self.directory.mkdir(exist_ok=True)
        self.keep = OnlineKeep(int(config.run.seed) + 91003,
            coefficient=float(config.f9.coefficient_override))
        self.factory = copy.copy(factory)
        self.factory._environments = {}
        # Independent TorchScript executors: prefill must not warm PPO's ray kernels.
        from ..environments.factory import SynthPointCloud
        module_name = 'diffusion_policy_3d.model.common._f9_teacher_synth'
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, inspect.getfile(SynthPointCloud))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        self.factory.synth_pc_class = sys.modules[module_name].SynthPointCloud
        self.teacher_task_sampler = str(config.f9.get('teacher_task_sampler', 'default'))
        self.blocked_sampler = None
        self.blocked_episodes = []
        self.sampler = None
        if self.teacher_task_sampler == 'blocked':
            from .f9_blocked_sampler import BlockedTeacherTaskSampler, library_tasks
            plan_path = Path(config.f9.experiment_root) / 'plan.json'
            self.blocked_sampler = BlockedTeacherTaskSampler(
                factory.tasks, self.factory, library_tasks(plan_path))
        else:
            self.sampler = TaskSampler(factory.tasks, int(config.run.seed) + 91001)
        with rng_scope(seed=int(config.run.seed) + 91002) as local:
            self.rng = capture_rng()
        self.ordinal = 0
        self.total_transitions = 0
        self.prefill_transitions = 0
        self.snapshot_path = None
        self.snapshot_sha = None
        self.setting = dict(d_safe=float(config.f9.d_safe), eta=float(config.f9.eta), M=int(config.f9.M))

    def collect(self, *, prefill=False, update=256):
        from scripts.routea_paired_support.core import write_json, write_npz, observation_hash
        from scripts.f7_analysis import first_strict_entry
        started = time.perf_counter()
        transitions, accepted_total, effective_total, triggered_total = 0, 0, 0, 0
        with torch.jit.optimized_execution(False), rng_scope(self.rng) as local:
            while (len(self.keep.records) < 64 and transitions < 4096) if prefill else (transitions < 1024):
                episode = self.ordinal
                picked = None
                if self.blocked_sampler is not None:
                    picked = self.blocked_sampler.sample()
                    task = picked['task']
                else:
                    task = self.sampler.sample()
                adapter = self.factory.make(task)
                env = adapter.environment
                env.record_execution_candidate = True
                reset_seed = int(hashlib.sha256(('f9:%s:%s:%s' % (
                    self.config.run.seed, episode, task.identity_hash)).encode()).hexdigest()[:8], 16)
                if picked is None:
                    observation, context = adapter.reset(seed=reset_seed)
                else:
                    observation, context = adapter.reset(
                        seed=reset_seed,
                        options=self.blocked_sampler.fixed_reset_options(picked))
                initial = env._drone_pos.copy()
                episode_dir = self.directory / 'episodes' / ('e%07d' % episode)
                resume_attempt = 1
                while episode_dir.exists():
                    resume_attempt += 1
                    episode_dir = self.directory/'episodes'/('e%07d_attempt%02d' % (episode,resume_attempt))
                episode_dir.mkdir(parents=True, exist_ok=False)
                header = dict(type='episode_start', episode=episode, update=update, task_id=task.task_id,
                    world_id=task.world_id, reset_seed=reset_seed, setting=self.setting,
                    initial_position=initial.tolist(), goal=env.goal_w.tolist(),
                    initial_yaw=float(env._drone_yaw), z_range=[float(env.z_min),float(env.z_max)],
                    spawn_offset=env._spawn_offset.tolist(), obstacle_path=task.obstacle_json,
                    obstacle_sha256=hashlib.sha256(Path(task.obstacle_json).read_bytes()).hexdigest(),
                    terminate_at_space_limits=bool(env.terminate_at_space_limits),
                    geometry=[asdict(o) for o in env.obstacles], heldout=episode % 10 == 0,
                    teacher_only=True, enters_ppo=False, resume_attempt=resume_attempt,
                    teacher_task_sampler=self.teacher_task_sampler,
                    teacher_task=(None if picked is None else dict(
                        start_pos=[float(v) for v in picked['start']],
                        start_yaw=float(picked['yaw']),
                        goal_pos=[float(v) for v in picked['goal']],
                        straight_min_clearance=float(picked['clearance']),
                        candidate_rejections=int(picked['rejections']))))
                records, paths, deltas = [], [], []
                with (episode_dir / 'trace.jsonl').open('x', encoding='utf-8') as stream:
                    import json
                    stream.write(json.dumps(header) + '\n')
                    for decision in range(60):
                        encoded = self.system.observation_adapter.encode(observation,
                            context['pointcloud_meta_history'], episode_identity_hash=context['episode_identity_hash'],
                            decision_index=context['decision_index'])
                        self.system.observation_adapter.verify(encoded)
                        pre, vel, yaw = env._drone_pos.copy(), env._drone_vel.copy(), float(env._drone_yaw)
                        cloud = torch.as_tensor(observation['point_cloud'][-1], dtype=torch.float32,
                                                device=self.system.bc_runtime.device)
                        pair = paired_sample(self.system.official_runtime, encoded.condition,
                            self.system.bc_runtime.policy.normalizer['action'].unnormalize,
                            cloud, pre, vel, yaw, self.setting, deterministic=False,
                            audit_noise=self.total_transitions + transitions == 0)
                        if pair['noise'] is not None:
                            torch.save(pair['noise'], self.directory / 'paired_noise_proof.pt')
                            write_json(self.directory / 'guidance_runtime_identity.json', pair['guidance_identity'])
                        next_obs, reward, terminated, truncated, info = adapter.step(ActionBundle.from_full(pair['full']))
                        if info['done_reason'] == 'invalid_state':
                            raise FloatingPointError('F9 teacher invalid_state')
                        path = np.asarray(info['executed_path'], np.float32)
                        records.append(dict(sample_id='e%07d_d%03d' % (episode, decision),
                            action=pair['normalized'][0].detach().cpu().float(),
                            condition=encoded.condition['state'][0].detach().cpu().float(),
                            observation={k: np.asarray(v, np.float32).copy() for k, v in observation.items()},
                            context=dict(pointcloud_meta_history=[asdict(m) for m in context['pointcloud_meta_history']],
                                episode_identity_hash=context['episode_identity_hash'], decision_index=context['decision_index']),
                            task_id=task.task_id, decision=decision, done_reason=info['done_reason'],
                            P6_condition_failed='不适用',
                            P6_label_status='not_applicable_online_random_condition'))
                        paths.append(path); deltas.append(pair['delta_exec'])
                        triggered_total += pair['triggered']; effective_total += pair['effective']
                        row = dict(type='transition', decision=decision, pre_state=dict(position=pre.tolist(),
                            velocity=vel.tolist(), yaw=yaw), full_action=pair['full'][0].cpu().tolist(),
                            base_full_action=pair['base_full'][0].cpu().tolist(),
                            normalized_target=pair['normalized'][0].cpu().tolist(),
                            observation_hash=observation_hash(observation), provenance=asdict(encoded.provenance),
                            delta_exec=pair['delta_exec'], effective=pair['effective'], triggered=pair['triggered'],
                            executed_path=path.tolist(), execution_candidate_path=info['execution_candidate_path'].tolist(),
                            event_segment=info['event_segment'],event_fraction=info['event_fraction'],
                            executed_time_s=info['executed_time_s'],
                            done_reason=info['done_reason'], reward=float(reward))
                        stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                        transitions += 1
                        observation, context = next_obs, info
                        if terminated or truncated:
                            break
                    accepted = self.keep.ingest(records, paths, env.goal_w, deltas, episode)
                    for record in accepted:
                        obs = record.pop('observation')
                        obs_path = episode_dir / (record['sample_id'] + '.npz')
                        write_npz(obs_path, obs)
                        record['observation_path'] = str(obs_path.resolve())
                        record['observation_sha256'] = hashlib.sha256(obs_path.read_bytes()).hexdigest()
                    # ingest copied dictionaries: replace with the finalized path-only records.
                    if episode % 10 == 0:
                        if accepted: self.keep.heldout[-len(accepted):] = accepted
                    else:
                        by_id = {r['sample_id']: r for r in accepted}
                        self.keep.records = [by_id.get(r['sample_id'], r) for r in self.keep.records]
                    accepted_total += len(accepted)
                    summary = dict(type='episode_end', ordinal=episode, decisions=len(paths),
                        eligible=len(accepted), heldout=episode % 10 == 0,
                        reached_1m=any(first_strict_entry(p, env.goal_w) is not None for p in paths),
                        done_reason=info['done_reason'])
                    stream.write(json.dumps(summary) + '\n')
                write_json(episode_dir / 'COMMITTED.json', dict(trace_sha256=hashlib.sha256(
                    (episode_dir / 'trace.jsonl').read_bytes()).hexdigest(), summary=summary))
                if picked is not None:
                    self.blocked_episodes.append(dict(
                        episode=episode, task_id=task.task_id, world_id=str(task.world_id),
                        start=[float(v) for v in picked['start']],
                        start_yaw=float(picked['yaw']),
                        goal=[float(v) for v in picked['goal']],
                        straight_min_clearance=float(picked['clearance']),
                        candidate_rejections=int(picked['rejections']),
                        attempts=int(picked['attempts'])))
                self.ordinal += 1
        self.rng = local['after']
        self.total_transitions += transitions
        if prefill:
            self.prefill_transitions += transitions
            if len(self.keep.records) < 64:
                raise RuntimeError('F9 prefill stopped: fewer than 64 eligible training decisions')
        result = dict(transitions=transitions, accepted=accepted_total, buffer=len(self.keep.records),
            heldout=len(self.keep.heldout), triggered=triggered_total, effective=effective_total,
            seconds=time.perf_counter() - started, prefill=prefill,
            prefill_overflow=max(0, transitions - 4096) if prefill else 0,
            teacher_task_sampler=self.teacher_task_sampler)
        if self.blocked_sampler is not None:
            clearances = [r['straight_min_clearance'] for r in self.blocked_episodes]
            result['blocked_sampling'] = dict(
                accepted=self.blocked_sampler.accepted,
                candidate_attempts=self.blocked_sampler.candidate_attempts,
                acceptance_rate=self.blocked_sampler.acceptance_rate(),
                rejection_histogram=dict(self.blocked_sampler.rejection_histogram),
                clearance_min=float(np.min(clearances)) if clearances else None,
                clearance_max=float(np.max(clearances)) if clearances else None,
                episodes=self.blocked_episodes[-8:])
        return result

    def snapshot(self, update):
        path = self.directory / ('buffer_u%06d.pt' % update)
        attempt = 1
        while path.exists():
            attempt += 1
            path = self.directory/('buffer_u%06d_attempt%02d.pt' % (update,attempt))
        torch.save(dict(records=self.keep.records, heldout=self.keep.heldout), path)
        self.snapshot_path = str(path.resolve())
        self.snapshot_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def begin_update(self):
        self.before_parameters = {name: p.detach().clone()
                                 for name, p in self.system.lora_bundle.named_adapter_parameters()}

    def parameter_changes(self):
        return {name: float((p.detach() - self.before_parameters[name]).float().norm())
                for name, p in self.system.lora_bundle.named_adapter_parameters()}

    def state_dict(self):
        return dict(snapshot=self.snapshot_path, snapshot_sha256=self.snapshot_sha,
            keep=self.keep.state_dict(),
            sampler=(None if self.sampler is None else self.sampler.state_dict()),
            blocked_sampler=(None if self.blocked_sampler is None else self.blocked_sampler.state_dict()),
            teacher_task_sampler=self.teacher_task_sampler, rng=self.rng,
            ordinal=self.ordinal, total_transitions=self.total_transitions,
            prefill_transitions=self.prefill_transitions)

    def load_state_dict(self, state):
        path = Path(state['snapshot'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == state['snapshot_sha256']
        bank = torch.load(path, map_location='cpu', weights_only=True)
        self.keep.records, self.keep.heldout = bank['records'], bank['heldout']
        self.keep.load_state_dict(state['keep'])
        if self.sampler is not None:
            self.sampler.load_state_dict(state['sampler'])
        if self.blocked_sampler is not None:
            self.blocked_sampler.load_state_dict(state['blocked_sampler'])
        for key in ('rng', 'ordinal', 'total_transitions', 'prefill_transitions'):
            setattr(self, key, state[key])
        self.snapshot_path, self.snapshot_sha = str(path), state['snapshot_sha256']
