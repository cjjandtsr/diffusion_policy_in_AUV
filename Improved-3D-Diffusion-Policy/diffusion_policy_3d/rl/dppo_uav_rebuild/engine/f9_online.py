"""F9 isolated teacher sampling and online executed-prefix regression."""
from contextlib import contextmanager
import hashlib
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from .selective_keep import executed_weights, prefix_x0_loss


def capture_rng():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=(n[0], n[1].tolist(), n[2], n[3], n[4]),
                cpu=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def rng_scope(state=None, *, seed=None):
    before = capture_rng()
    holder = {}
    try:
        if state is not None:
            restore_rng(state)
        elif seed is not None:
            random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield holder
    finally:
        holder['after'] = capture_rng()
        restore_rng(before)


def execution_delta(base, guided, position, velocity, yaw):
    from diffusion_policy_3d.env.traj_utils import action_to_world
    a = action_to_world(np.asarray(base, np.float32)[2:6], np.asarray(position, np.float32),
                        np.asarray(velocity, np.float32), yaw)[:, :3]
    b = action_to_world(np.asarray(guided, np.float32)[2:6], np.asarray(position, np.float32),
                        np.asarray(velocity, np.float32), yaw)[:, :3]
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1).max())


def eligible_indices(paths, goal, deltas, reasons=None):
    from scripts.f7_analysis import first_strict_entry
    dstar = None
    for i, p in enumerate(paths):
        if first_strict_entry(np.asarray(p, float), np.asarray(goal, np.float32)) is not None:
            dstar = i
            break
        if reasons and reasons[i] in ('collision', 'ground_crash', 'invalid_state'):
            break
    return [] if dstar is None else [i for i in range(dstar) if deltas[i] >= .05]


def stage_gate(d_t, d_c, arrived_t, arrived_c, r, *, restarted=False):
    t = Fraction(*d_t) if isinstance(d_t, tuple) else Fraction(str(d_t))
    c = Fraction(*d_c) if isinstance(d_c, tuple) else Fraction(str(d_c))
    if t <= Fraction(7, 10) * c:
        return 'pass' if arrived_t >= arrived_c - 20 else 'behavior_changed_closed_loop_worse'
    if r is None or not np.isfinite(r):
        return 'pending_r'
    if r > .5:
        return 'stop_after_restart' if restarted else 'restart'
    return 'fitted_without_student_change'


def demand_counts(paths, goal, deltas):
    from scripts.f7_analysis import first_strict_entry
    dstar = next((i for i,p in enumerate(paths) if first_strict_entry(
        np.asarray(p,float),np.asarray(goal,np.float32)) is not None),None)
    denominator = len(paths) if dstar is None else dstar
    return sum(d >= .05 for d in deltas[:denominator]), denominator, dstar


def validate_teacher_trace(header, rows, summary):
    """Recompute first events on the RECORDED stochastic execution candidate."""
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec
    from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
    from diffusion_policy_3d.env.traj_utils import action_to_world
    obstacles = [ObstacleSpec(**o) for o in header['geometry']]
    previous = np.asarray(header['initial_position'], np.float32)
    for i, row in enumerate(rows):
        assert row['decision'] == i
        pre = row['pre_state']
        assert np.allclose(previous, pre['position'], atol=1e-6, rtol=0)
        wp = action_to_world(np.asarray(row['full_action'],np.float32)[2:10],
            np.asarray(pre['position'],np.float32),np.asarray(pre['velocity'],np.float32),pre['yaw'])
        candidate = np.asarray(row['execution_candidate_path'],np.float32)
        # Noise/disturbance may alter only the final candidate point.
        assert np.allclose(candidate[0],previous,atol=1e-6,rtol=0)
        assert np.allclose(candidate[1:-1],wp[:len(candidate)-2,:3],atol=1e-5,rtol=0)
        event = locate_first_event(candidate,np.asarray(header['goal'],np.float32),obstacles,
            .1,header['z_range'][1],terminate_at_space_limits=header['terminate_at_space_limits'])
        if event is None:
            expected = candidate; seconds = 1.2
            assert row['done_reason'] in ('none','time_limit')
            assert row['done_reason']!='time_limit' or i==59
        else:
            expected = np.vstack((candidate[:event.segment_index+1],event.position.astype(np.float32)))
            seconds = (event.segment_index+event.fraction)*.3
            assert row['done_reason']==event.kind and i==len(rows)-1
            assert row['event_segment']==event.segment_index
            assert abs(row['event_fraction']-event.fraction)<1e-6
        assert np.allclose(row['executed_path'],expected,atol=1e-5,rtol=0)
        assert abs(row['executed_time_s']-seconds)<1e-6
        expected_delta = execution_delta(row['base_full_action'],row['full_action'],
            pre['position'],pre['velocity'],pre['yaw'])
        assert abs(expected_delta-row['delta_exec'])<1e-7
        assert row['effective']==(expected_delta>=.05)
        previous = expected[-1]
    assert rows and summary['decisions']==len(rows) and summary['done_reason']==rows[-1]['done_reason']
    assert rows[-1]['done_reason']!='none'
    eligible = eligible_indices([r['executed_path'] for r in rows],header['goal'],
        [r['delta_exec'] for r in rows],[r['done_reason'] for r in rows])
    assert summary['eligible']==len(eligible)
    return eligible


def paired_sample(runtime, condition, unnormalize, cloud, position, velocity, yaw,
                  setting, *, deterministic, audit_noise=False):
    """Two complete DDIM chains, same random stream; advance it exactly once."""
    from scripts.routea_denoising_guidance import Guidance, install
    from scripts.run_routea_xk_evaluation import load_override
    root = Path(__file__).resolve().parents[4]
    guide = Guidance(unnormalize, setting['eta'], 0, setting['d_safe'], setting['M'])
    guide.begin(cloud, position, yaw)
    noise = [[], []]

    def invoke(which):
        original = (torch.randn, torch.randn_like)
        if audit_noise:
            def record(fn):
                def call(*a, **kw):
                    value = fn(*a, **kw)
                    noise[which].append(value.detach().cpu().clone())
                    return value
                return call
            torch.randn, torch.randn_like = map(record, original)
        try:
            with torch.no_grad():
                return runtime(cond=condition, deterministic=deterministic,
                               return_chain=True, use_base_policy=False).trajectories.detach()
        finally:
            if audit_noise:
                torch.randn, torch.randn_like = original

    mode = 'deterministic' if deterministic else 'random'
    with load_override(root)(runtime, denoised_clip_value=2., sampling_mode=mode,
                             min_std_vector=[.1]*11 if deterministic else None):
        before = capture_rng()
        base = invoke(0)
        after = capture_rng()
        restore_rng(before)
        original_mean = runtime.p_mean_var
        try:
            identity = install(runtime, guide)
            guided = invoke(1)
            other = capture_rng()
            assert torch.equal(after['cpu'], other['cpu'])
            assert len(after['cuda']) == len(other['cuda'])
            assert all(torch.equal(a, b) for a, b in zip(after['cuda'], other['cuda']))
            assert after['numpy'] == other['numpy'] and after['python'] == other['python']
        finally:
            runtime.p_mean_var = original_mean
            restore_rng(after)
    if audit_noise:
        assert len(noise[0]) == len(noise[1]) and len(noise[0]) > 0
        assert all(torch.equal(a, b) for a, b in zip(*noise))
    full = unnormalize(guided)
    base_full = unnormalize(base)
    delta = execution_delta(base_full[0].cpu().numpy(), full[0].cpu().numpy(), position, velocity, yaw)
    return dict(normalized=guided, full=full, base_full=base_full, delta_exec=delta,
                effective=delta >= .05, triggered=bool(guide.triggered_steps),
                noise=noise if audit_noise else None, guidance_identity=identity)


class OnlineKeep:
    def __init__(self, seed, *, capacity=4096, batch_size=32, coefficient=0.):
        self.generator = torch.Generator().manual_seed(int(seed))
        self.coefficient = float(coefficient)
        self.capacity, self.batch_size = capacity, batch_size
        self.records, self.heldout = [], []

    def ingest(self, records, paths, goal, deltas, episode_number):
        indices = eligible_indices(paths, goal, deltas, [r.get('done_reason', 'none') for r in records])
        accepted = []
        for j, i in enumerate(indices):
            record = dict(records[i], intervention='first_intervention' if j == 0 else 'after_intervention',
                          episode_number=episode_number, delta_exec=float(deltas[i]))
            accepted.append(record)
        if episode_number % 10 == 0:
            self.heldout.extend(accepted)
        else:
            self.records.extend(accepted)
            self.records = self.records[-self.capacity:]
        return accepted

    def loss(self, runtime, *, device):
        ids = torch.randint(len(self.records), (self.batch_size,), generator=self.generator).tolist()
        selected = [self.records[i] for i in ids]
        x0 = torch.stack([r['action'] for r in selected]).to(device)
        cond = torch.stack([r['condition'] for r in selected]).to(device)
        weights = executed_weights('none').expand(len(ids), -1, -1).to(device)
        times = runtime.ddim_t[-runtime.ft_denoising_steps:].detach().cpu().long()
        slots = torch.randint(len(times), (len(ids),), generator=self.generator)
        noise = torch.randn(x0.shape, generator=self.generator).to(device)
        alpha = runtime.ddim_alphas[-len(times):][slots.to(runtime.ddim_alphas.device)].to(device)
        return prefix_x0_loss(runtime, x0, cond, weights, times[slots].to(device), alpha, noise).mean()

    def state_dict(self):
        return dict(coefficient=self.coefficient, rng=self.generator.get_state().clone())

    def load_state_dict(self, state):
        self.coefficient = float(state['coefficient'])
        self.generator.set_state(state['rng'].cpu())
