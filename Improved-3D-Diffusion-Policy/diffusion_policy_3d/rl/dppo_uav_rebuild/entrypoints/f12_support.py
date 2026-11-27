"""F12 phase 2 (Claude, 2026-11-26): registry, validation and the RX sidecar.

Arms (all fork P6@u256, 64 updates u257..u320, same seed within a batch):
  C0  reward v2 unchanged                                   (matched control)
  R   reward v6 = v2 + newbest direction gate + collision -100 + stall
  RN  R + sampling/likelihood std floor 0.1 -> 0.2           (exploratory arm)
  RX  R + condition-perturbation candidate groups scored by one-step critic TD

Design: 研究报告/2026-11-26_F12方案变更对照与第二阶段定稿_v1_3.md and
执行指引/2026-11-26_F12第二阶段_代码说明与WSL执行指引_v1_0.md.
"""
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .f7_support import KEEP, REFERENCE_SHA

SEEDS = (202611261, 202611262)
RUNS = {'P39': ('C0', SEEDS[0]), 'P40': ('R', SEEDS[0]), 'P41': ('RN', SEEDS[0]), 'P42': ('RX', SEEDS[0]),
        'P43': ('C0', SEEDS[1]), 'P44': ('R', SEEDS[1]), 'P45': ('RN', SEEDS[1]), 'P46': ('RX', SEEDS[1])}
FLOOR = {'C0': .1, 'R': .1, 'RN': .2, 'RX': .1}
RX_DEFAULTS = dict(groups_per_update=16, transition_cap=1024, states_per_episode=2, min_spacing=3,
                   free_min=3.0, free_max=6.0, goal_margin=0.5,
                   rotations_deg=[30.0, -30.0, 60.0, -60.0], mirror=True, minibatch_pairs=64,
                   std_epsilon=1e-6, coefficient_rule='candidates_over_1024')


def reward_params():
    from diffusion_policy_3d.env.reward_v6 import DEFAULTS
    return dict(DEFAULTS)


def validate_f12(config):
    f, f7 = config.f12, config.get('f7')
    arm, seed = RUNS[str(f.run_id)]
    if set(f) != {'run_id', 'arm', 'reward', 'rx'} or str(f.arm) != arm:
        raise ValueError('F12 block differs from registry')
    if f7 is None or set(f7) != {'run_id', 'approach_weight', 'adaptive_lr', 'keep_coefficient',
                                 'reference_parent_receipt_sha256'}:
        raise ValueError('F12 reuses the F7 P6-fork block')
    if (str(f7.run_id) != str(f.run_id) or bool(f7.adaptive_lr) or float(f7.approach_weight) != 1.
            or float(f7.keep_coefficient) != KEEP):
        raise ValueError('F12 F7-block values differ from registry')
    expected_reward = {} if arm == 'C0' else reward_params()
    if dict(f.reward) != expected_reward:
        raise ValueError('F12 reward parameters differ from registered R12prime')
    if str(config.reward_version) != ('v2' if arm == 'C0' else 'v6'):
        raise ValueError('F12 reward version differs from arm')
    rx = dict(f.rx)
    if bool(rx.pop('enabled')) != (arm == 'RX') or (arm == 'RX' and rx != RX_DEFAULTS):
        raise ValueError('F12 RX settings differ from registry')
    d = config.dppo
    if (float(d.min_sampling_std) != FLOOR[arm] or float(d.min_logprob_std) != FLOOR[arm]
            or int(config.run.seed) != seed or int(config.run.updates) != 320
            or int(config.run.rollout_transitions) != 1024
            or float(config.optimizer.actor_lr) != 1e-5 or float(config.optimizer.actor_lr_end) != 1e-5
            or float(config.optimizer.critic_lr) != 1e-3
            or config.a2.reference_sha256 != REFERENCE_SHA or int(config.a2.keep_seed) != 2026102301
            or config.a2.arm != 'T' or float(config.task_protocol.space_cost_budget) != 1.
            or any(k in config for k in ('f5', 'f6', 'f8', 'f9', 'xk', 'task_bank'))
            or set(d) & {'ppo_inert_mask', 'self_imitation', 'inert_anchor', 'min_std_vector',
                         'denoised_clip_value', 'drift_monitor', 'drift_gate'}):
        raise ValueError('F12 differs from registered phase-2 settings')


from ..engine.f12_rx_math import rotate_goal, mirror_full, group_advantages, td_score, horizontal_free  # noqa: F401


class _Batch:
    """Minimal pair batch accepted by compute_official_components."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class RXSession:
    """Collect candidate groups before each PPO phase; expose an actor-only clipped loss.

    Plugged into Trainer.f8_keep (gradient added after the PPO gradient, times
    `coefficient`).  Records live for one outer update only.
    """

    def __init__(self, system, config, factory, results):
        from ..environments.task_sampler import TaskSampler
        from ..engine.f9_online import rng_scope, capture_rng
        self.system, self.config = system, config
        self.spec = {**RX_DEFAULTS, **{k: v for k, v in dict(config.f12.rx).items() if k != 'enabled'}}
        self.directory = Path(results) / 'f12_rx'
        self.directory.mkdir(exist_ok=True)
        self.factory = copy.copy(factory)
        self.factory._environments = {}
        from ..environments.factory import SynthPointCloud
        name = 'diffusion_policy_3d.model.common._f12_rx_synth'
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, inspect.getfile(SynthPointCloud))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.factory.synth_pc_class = sys.modules[name].SynthPointCloud
        self.sampler = TaskSampler(factory.tasks, int(config.run.seed) + 120001)
        with rng_scope(seed=int(config.run.seed) + 120002):
            self.rng = capture_rng()
        self.generator = torch.Generator().manual_seed(int(config.run.seed) + 120003)
        self.coefficient = 0.
        self.records = []
        self.ordinal = 0
        self.total_transitions = 0
        self.finite_task = config.get('task_protocol') is not None

    # -- helpers -----------------------------------------------------------
    def _critic_condition(self, condition, env):
        from ..engine.critic_state import with_history_scalar, with_episode_context
        return (with_episode_context(condition, env) if self.finite_task
                else with_history_scalar(condition, env._reward_history))

    def _encode_raw(self, agent_pos, point_cloud):
        from ..actor_ddpm.conditioning import encode_e5_condition_with_normalized
        dev = self.system.bc_runtime.device
        po = {'agent_pos': torch.as_tensor(np.asarray(agent_pos, np.float32)[None], device=dev),
              'point_cloud': torch.as_tensor(np.asarray(point_cloud, np.float32)[None], device=dev)}
        cond, _ = encode_e5_condition_with_normalized(self.system.bc_runtime.policy, po,
                                                      self.system.observation_adapter.spec)
        return cond

    def _clone(self, adapter):
        env = adapter.environment
        memo = {id(env.synth_pc): env.synth_pc}
        new = copy.copy(adapter)
        new.environment = copy.deepcopy(env, memo)
        new.reward_state = copy.deepcopy(adapter.reward_state)
        return new

    def _true_logprobs(self, condition, chain):
        rt = self.system.official_runtime
        flat = rt.get_logprobs(condition, chain)
        return flat.reshape(chain.shape[0], rt.ft_denoising_steps, chain.shape[2], chain.shape[3])

    def _candidates(self, encoded, observation, env):
        """Return list of (label, normalized_chain (1,11,16,11), full (1,16,11))."""
        sampler, norm = self.system.rollout_sampler, self.system.bc_runtime.policy.normalizer['action']
        out = []
        base = sampler.sample(encoded.condition, deterministic=False)
        out.append(('orig', base.normalized_chain, base.action.full))
        for deg in self.spec['rotations_deg']:
            cond = self._encode_raw(rotate_goal(observation['agent_pos'], math.radians(float(deg))),
                                    observation['point_cloud'])
            r = sampler.sample(cond, deterministic=False)
            out.append(('rot%+d' % int(deg), r.normalized_chain, r.action.full))
        if self.spec['mirror']:
            m = sampler.sample(encoded.condition, deterministic=False)
            chain_full = norm.unnormalize(m.normalized_chain)
            mirrored = torch.as_tensor(mirror_full(chain_full.detach().cpu().numpy(), float(env._drone_yaw)),
                                       dtype=chain_full.dtype, device=chain_full.device)
            out.append(('mirror', norm.normalize(mirrored), mirrored[:, -1]))
        return out

    # -- collection --------------------------------------------------------
    def collect(self, update):
        from ..engine.f9_online import rng_scope
        from ..engine.contracts import ActionBundle
        from .pipeline import _bootstrap_allowed
        started = time.perf_counter()
        self.records, groups, skipped, transitions, episodes = [], [], 0, 0, 0
        gamma = float(self.config.dppo.gamma_environment)
        with torch.jit.optimized_execution(False), torch.no_grad(), rng_scope(self.rng) as local:
            while len(groups) < int(self.spec['groups_per_update']) and transitions < int(self.spec['transition_cap']):
                task = self.sampler.sample()
                adapter = self.factory.make(task)
                env = adapter.environment
                reset_seed = int(hashlib.sha256(('f12rx:%s:%s:%s' % (
                    self.config.run.seed, self.ordinal, task.identity_hash)).encode()).hexdigest()[:8], 16)
                observation, context = adapter.reset(seed=reset_seed)
                taken, last = 0, -99
                for decision in range(int(env.max_episode_steps)):
                    encoded = self.system.observation_adapter.encode(
                        observation, context['pointcloud_meta_history'],
                        episode_identity_hash=context['episode_identity_hash'],
                        decision_index=context['decision_index'])
                    self.system.observation_adapter.verify(encoded)
                    pos, goal = env._drone_pos.astype(np.float64), np.asarray(env.goal_w, np.float64)
                    free = horizontal_free(pos, goal, env.obstacles)
                    eligible = (taken < int(self.spec['states_per_episode'])
                                and decision - last >= int(self.spec['min_spacing'])
                                and float(self.spec['free_min']) <= free <= float(self.spec['free_max'])
                                and float(np.linalg.norm((goal - pos)[:2])) > free + float(self.spec['goal_margin'])
                                and len(groups) < int(self.spec['groups_per_update']))
                    if eligible:
                        critic_cond = self._critic_condition(encoded.condition, env)
                        v0 = float(self.system.critic(critic_cond).reshape(-1)[0].item())
                        group = []
                        cands = self._candidates(encoded, observation, env)
                        for label, chain, full in cands:
                            branch = self._clone(adapter)
                            nobs, rew, term, trunc, info = branch.step(ActionBundle.from_full(full))
                            if info['done_reason'] == 'invalid_state':
                                raise FloatingPointError('F12 RX candidate invalid_state')
                            boot = _bootstrap_allowed(term, trunc, finite_task=self.finite_task)
                            v1 = 0.
                            if boot:
                                nenc = self.system.observation_adapter.encode(
                                    nobs, info['pointcloud_meta_history'],
                                    episode_identity_hash=info['episode_identity_hash'],
                                    decision_index=info['decision_index'])
                                v1 = float(self.system.critic(self._critic_condition(
                                    nenc.condition, branch.environment)).reshape(-1)[0].item())
                            group.append(dict(label=label, chain=chain[0].detach().cpu(),
                                old_logprobs=self._true_logprobs(encoded.condition, chain)[0].detach().cpu(),
                                reward=float(rew), v_after=v1, bootstrap=bool(boot),
                                score=td_score(rew, v0, v1, boot, gamma),
                                done_reason=info['done_reason'],
                                end=np.asarray(info['executed_path'][-1], np.float64).tolist()))
                        adv, keep = group_advantages([g['score'] for g in group], float(self.spec['std_epsilon']))
                        cond_cpu = {k: v[0].detach().cpu() for k, v in critic_cond.items()}
                        summary = dict(update=update, episode=self.ordinal, decision=decision, task_id=task.task_id,
                                       world_id=str(task.world_id), free_along_goal=free, v_before=v0, kept=keep,
                                       candidates=[dict({k: g[k] for k in ('label', 'reward', 'v_after', 'bootstrap',
                                                    'score', 'done_reason', 'end')}, advantage=float(a))
                                                   for g, a in zip(group, adv)])
                        groups.append(summary)
                        if keep:
                            for g, a in zip(group, adv):
                                self.records.append(dict(condition=cond_cpu, chain=g['chain'],
                                                         old_logprobs=g['old_logprobs'], advantage=float(a),
                                                         label=g['label']))
                        else:
                            skipped += 1
                        taken, last = taken + 1, decision
                    rollout = self.system.rollout_sampler.sample(encoded.condition, deterministic=False)
                    observation, _, term, trunc, context = adapter.step(rollout.action)
                    transitions += 1
                    if term or trunc or transitions >= int(self.spec['transition_cap']):
                        break
                episodes += 1
                self.ordinal += 1
        self.rng = local['after']
        self.total_transitions += transitions
        n = len(self.records)
        self.coefficient = n / 1024. if n else 0.
        result = dict(update=update, groups=len(groups), kept_groups=len(groups) - skipped,
                      skipped_equal_scores=skipped, candidate_records=n, coefficient=self.coefficient,
                      sidecar_transitions=transitions, sidecar_episodes=episodes,
                      seconds=time.perf_counter() - started, groups_detail=groups)
        return result

    # -- loss --------------------------------------------------------------
    def loss(self, runtime, *, device):
        from ..engine.official_loss_components import AdvantageStats, compute_official_components
        m = min(int(self.spec['minibatch_pairs']), len(self.records))
        ids = torch.randint(len(self.records), (m,), generator=self.generator).tolist()
        k = torch.randint(0, int(runtime.ft_denoising_steps), (m,), generator=self.generator)
        recs = [self.records[i] for i in ids]
        chains = torch.stack([r['chain'] for r in recs])
        rows = torch.arange(m)
        batch = _Batch(
            condition={key: torch.stack([r['condition'][key] for r in recs]).to(device) for key in recs[0]['condition']},
            chains_prev=chains[rows, k].to(device), chains_next=chains[rows, k + 1].to(device),
            denoising_inds=k.to(device),
            advantages=torch.tensor([r['advantage'] for r in recs], dtype=torch.float32, device=device),
            oldlogprobs=torch.stack([r['old_logprobs'] for r in recs])[rows, k].to(device),
            returns=torch.zeros(m, device=device), oldvalues=torch.zeros(m, device=device))
        one = torch.ones((), device=device)
        stats = AdvantageStats(mean=0. * one, std=one - 1e-8, lower_quantile=-math.inf * one,
                               upper_quantile=math.inf * one)
        parts = compute_official_components(runtime, batch, stats, reward_start=2, reward_horizon=8,
                                            use_bc_loss=False, inert_mask=None)
        return parts.policy.mean()

    # -- checkpoint state --------------------------------------------------
    def state_dict(self):
        return dict(rng=self.rng, generator=self.generator.get_state().clone(), ordinal=self.ordinal,
                    total_transitions=self.total_transitions, sampler=self.sampler.state_dict())

    def load_state_dict(self, state):
        self.rng, self.ordinal = state['rng'], state['ordinal']
        self.total_transitions = state['total_transitions']
        self.generator.set_state(state['generator'].cpu())
        self.sampler.load_state_dict(state['sampler'])
