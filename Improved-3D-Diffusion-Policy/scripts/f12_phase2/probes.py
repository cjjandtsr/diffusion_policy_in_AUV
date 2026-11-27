"""F12 phase-2 behavioural probes (Claude 2026-11-26). Expected values are hand-computed.

  python scripts/f12_phase2/probes.py cpu --prepared <dir with prepared.json> --out <json>
  python scripts/f12_phase2/probes.py gpu --out <json>        # P6@u256 native restore, GPU

Each probe prints `PROBE <id> PASS|FAIL expected=... actual=...`; exit code = number of FAILs.
"""
import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = []


def check(pid, ok, expected, actual, note=''):
    RESULTS.append(dict(id=pid, passed=bool(ok), expected=str(expected), actual=str(actual), note=note))
    print('PROBE %s %s expected=%s actual=%s %s' % (pid, 'PASS' if ok else 'FAIL', expected, actual, note), flush=True)


def ob(**k):
    base = dict(shape='cylinder', x=0., y=0., z=0., radius=0., height=0., size_x=0., size_y=0., size_z=0., yaw=0.)
    base.update(k)
    return types.SimpleNamespace(**base)


class FakeEnv:
    """Minimal stand-in to drive step_v6 around a stubbed step_v2."""
    def __init__(self, obstacles, goal):
        self.obstacles, self.goal_w, self.episode_step = obstacles, np.asarray(goal, float), 0


def run_v6(env, path, reason, newbest, terminal):
    import diffusion_policy_3d.env.reward_v2_support.environment as e2
    from diffusion_policy_3d.env import reward_v6

    def fake(env_, action):
        env_.episode_step += 1
        terms = dict(terminal=terminal, newbest=newbest, risk=-0.1, time=-.05, space=-.01)
        return None, sum(terms.values()), reason != 'none', False, dict(
            reward_terms=terms, reward_metrics={}, done_reason=reason, executed_path=np.asarray(path, float))
    original = e2.step_v2
    e2.step_v2 = fake
    try:
        return reward_v6.step_v6(env, None)
    finally:
        e2.step_v2 = original


def cpu(args):
    from diffusion_policy_3d.env import reward_v6 as r6
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine import f12_rx_math as rx
    cyl = ob(shape='cylinder', x=3.8, y=0., z=2., radius=.5, height=4.)   # surface at x = 3.3
    # A1 gate: obstacle surface 3.3 m ahead -> clearance < 0.3 from t = 3.0 m -> D_f = 3.0, g = 0.5
    D = r6.forward_free_distance([0, 0, 1.5], [1, 0, 0], [cyl])
    check('A1a_free_ahead', abs(D - 3.0) < 1e-9, 3.0, D)
    check('A1b_gate_ahead', abs(r6.direction_gate(D) - .5) < 1e-9, .5, r6.direction_gate(D))
    # A1 negative: pass parallel at 1.2 m lateral -> nothing within 10 m -> g = 1
    D = r6.forward_free_distance([0, 1.2, 1.5], [1, 0, 0], [cyl])
    check('A1c_parallel_gate1', r6.direction_gate(D) == 1., 1., r6.direction_gate(D))
    # A1 floor: obstacle 0.5 m ahead -> D_f 0.2... g clipped to 0.2
    D = r6.forward_free_distance([2.6, 0, 1.5], [1, 0, 0], [cyl])
    check('A1d_gate_floor', r6.direction_gate(D) == .2, .2, (D, r6.direction_gate(D)))
    # A1 near goal with an obstacle BEHIND the goal: start (2.0,0,1.5), goal (3.0,0,1.5)? goal must be
    # >=1.63 m from obstacles: put goal at x=1.0 (2.3 m from surface); start x=0 moving +x.
    env = FakeEnv([cyl], [1.0, 0., 1.5])
    _, total, *_rest, info = run_v6(env, [[0, 0, 1.5], [.3, 0, 1.5], [.6, 0, 1.5]], 'none', 2.0, 0.)
    m, t = info['reward_metrics'], info['reward_terms']
    check('A1e_goal_before_obstacle_gate1', m['f12_gate'] == 1. and t['newbest'] == 2.0,
          'gate 1, newbest 2.0 (D_f 3.0 > goal 1.0)', (m['f12_gate'], t['newbest'], m['f12_free_distance']))
    env = FakeEnv([cyl], [-8.0, 0., 1.5])   # goal far behind start: obstacle ahead, goal not before it
    _, total, *_rest, info = run_v6(env, [[0, 0, 1.5], [.3, 0, 1.5], [.6, 0, 1.5]], 'none', 2.0, 0.)
    check('A1f_gated_newbest', abs(info['reward_terms']['newbest'] - 1.0) < 1e-9, 1.0, info['reward_terms']['newbest'])
    check('A1g_total_sum', abs(total - sum(info['reward_terms'].values())) < 1e-12, 'sum(terms)', total)
    # A2 terminal mapping
    for reason, term, exp in (('collision', -50., -100.), ('ground_crash', -50., -100.), ('out_of_bounds', -50., -50.),
                              ('altitude_exceed', -50., -50.), ('reached', 100., 100.), ('time_limit', 0., 0.)):
        env = FakeEnv([], [50., 0., 1.5])
        _, _, *_r, info = run_v6(env, [[0, 0, 1.5], [.3, 0, 1.5]], reason, 0., term)
        check('A2_terminal_' + reason, info['reward_terms']['terminal'] == exp, exp, info['reward_terms']['terminal'])
    # A3 stall: five decisions of 0.05 m -> sum 0.25 < 0.3 -> -0.2 at the 5th; 0.05*4 + 0.2 = 0.4 -> 0
    env = FakeEnv([], [50., 0., 1.5]); out = []
    for i in range(6):
        _, _, *_r, info = run_v6(env, [[i * .05, 0, 1.5], [i * .05 + .05, 0, 1.5]], 'none', 0., 0.)
        out.append(info['reward_terms']['stall'])
    check('A3a_stall_trigger', out == [0., 0., 0., 0., -.2, -.2], [0, 0, 0, 0, -.2, -.2], out)
    env = FakeEnv([], [50., 0., 1.5]); out = []
    for step in (.05, .05, .05, .05, .2):
        _, _, *_r, info = run_v6(env, [[0, 0, 1.5], [step, 0, 1.5]], 'none', 0., 0.)
        out.append(info['reward_terms']['stall'])
    check('A3b_stall_no_trigger', out == [0.] * 5, [0.] * 5, out)
    env = FakeEnv([], [50., 0., 1.5])
    for i in range(4):
        run_v6(env, [[0, 0, 1.5], [.01, 0, 1.5]], 'none', 0., 0.)
    _, _, *_r, info = run_v6(env, [[0, 0, 1.5], [.01, 0, 1.5]], 'collision', 0., -50.)
    check('A3c_no_stall_on_collision', info['reward_terms']['stall'] == 0., 0., info['reward_terms']['stall'])
    # A6 RX math
    r = rx.td_score(1.5, 2.0, 3.0, True)
    check('A6a_td_bootstrap', abs(r - 2.47) < 1e-12, 2.47, r)
    r = rx.td_score(1.5, 2.0, 3.0, False)
    check('A6b_td_terminal_V0', abs(r + .5) < 1e-12, -.5, r)
    a, keep = rx.group_advantages([1., 2., 3.])
    check('A6c_group_norm', keep and np.allclose(a, [-1.224744871, 0., 1.224744871]), '[-1.2247,0,1.2247]', a)
    a, keep = rx.group_advantages([2., 2., 2.])
    check('A6d_all_equal_zero', (not keep) and not np.any(a), 'zeros, skipped', (a, keep))
    full = np.random.default_rng(0).normal(size=(16, 11)).astype(np.float32)
    twice = rx.mirror_full(rx.mirror_full(full, .7), .7)
    check('A6e_mirror_involution', np.allclose(twice, full, atol=1e-5), 'identity', float(np.abs(twice - full).max()))
    v = np.zeros((16, 11), np.float32); v[:, 0] = 1.   # body x forward at yaw 0 stays; body y flips
    v[:, 1] = .5
    mv = rx.mirror_full(v, 0.)
    check('A6f_mirror_lateral', np.allclose(mv[:, 0], 1.) and np.allclose(mv[:, 1], -.5), 'x 1, y -0.5', mv[0, :2])
    v2 = np.zeros((16, 11), np.float32); v2[:, 3], v2[:, 4], v2[:, 6], v2[:, 7] = 2., .3, .1, -.4
    m2 = rx.mirror_full(v2, 0.)
    check('A6f2_mirror_vel_acc', np.allclose(m2[:, 3:5], [2., -.3]) and np.allclose(m2[:, 6:8], [.1, .4]),
          'vel (2,-0.3) acc (0.1,0.4)', (m2[0, 3:5], m2[0, 6:8]))
    v3 = np.zeros((16, 11), np.float32); v3[:, 0] = 1.   # yaw 90 deg: world +x is body right (-y)
    m3 = rx.mirror_full(v3, math.pi / 2)
    check('A6f3_mirror_yaw90', np.allclose(m3[:, 0], -1., atol=1e-6) and np.allclose(m3[:, 1], 0., atol=1e-6),
          'world (1,0) -> (-1,0) at yaw 90', m3[0, :2])
    ap = np.zeros((3, 9), np.float32); ap[:, 3] = 4.; ap[:, 4] = 0.
    rot = rx.rotate_goal(ap, math.radians(30))
    check('A6g_rotate_goal', np.allclose(rot[:, 3], 4 * math.cos(math.radians(30))) and np.allclose(rot[:, 4], 2.),
          '(3.464, 2.0)', rot[0, 3:5])
    check('A6h_rotate_other_dims_untouched', np.array_equal(np.delete(rot, [3, 4], 1), np.delete(ap, [3, 4], 1)), 'equal', 'checked')
    # A7 registry / config validation of the prepared configs
    if args.prepared:
        from omegaconf import OmegaConf
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
        prepared = json.loads((Path(args.prepared) / 'prepared.json').read_text())
        for name, item in prepared['configs'].items():
            c = OmegaConf.load(item['path'])
            validate_v2_config(c)
            check('A7_config_' + name, True, 'validates', '%s arm=%s reward=%s floor=%s/%s rx=%s seed=%s' % (
                name, c.f12.arm, c.reward_version, c.dppo.min_sampling_std, c.dppo.min_logprob_std,
                c.f12.rx.enabled, c.run.seed))
            bad = OmegaConf.create(OmegaConf.to_container(c))
            bad.dppo.min_sampling_std = .3
            try:
                validate_v2_config(bad); ok = False
            except ValueError:
                ok = True
            check('A7_negative_floor_' + name, ok, 'ValueError for floor 0.3', ok)


def gpu(args):
    import torch
    from scripts.f12_phase1 import f12_common as C
    system, config, header, _ = C.load_system()
    spec = __import__('importlib.util').util.spec_from_file_location(
        'f12_rx_math', ROOT / 'diffusion_policy_3d/rl/dppo_uav_rebuild/engine/f12_rx_math.py')
    rx = __import__('importlib.util').util.module_from_spec(spec); spec.loader.exec_module(rx)
    from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized
    z = np.load(C.OUT / 's0' / 's0_states.npz', allow_pickle=False)
    idx = np.where(z['kind'].astype(str) == 'A')[0][:8]
    dev = system.bc_runtime.device
    rt, norm = system.official_runtime, system.bc_runtime.policy.normalizer['action']

    def encode(agent, pc):
        po = {'agent_pos': torch.as_tensor(np.asarray(agent, np.float32)[None], device=dev),
              'point_cloud': torch.as_tensor(np.asarray(pc, np.float32)[None], device=dev)}
        return encode_e5_condition_with_normalized(system.bc_runtime.policy, po, system.observation_adapter.spec)[0]
    i = int(idx[0]); agent, pc, yaw = z['agent_pos'][i], z['point_cloud'][i], float(z['pre_yaw'][i])
    c0 = encode(agent, pc)
    c0b = encode(rx.rotate_goal(agent, 0.), pc)
    check('G1_rotate0_same_condition', all(torch.equal(c0[k], c0b[k]) for k in c0), 'identical', 'compared')
    c30 = encode(rx.rotate_goal(agent, math.radians(30)), pc)
    check('G1b_rotate30_changes', any(not torch.equal(c0[k], c30[k]) for k in c0), 'different', 'compared')
    torch.manual_seed(5)
    with torch.no_grad():
        s = rt(cond=c0, deterministic=False, return_chain=True, use_base_policy=False)
        chain = s.chains
        check('G2_chain_last_is_action', torch.allclose(chain[:, -1], s.trajectories, atol=1e-6), 'equal',
              float((chain[:, -1] - s.trajectories).abs().max()))
        lp_a = rt.get_logprobs(c0, chain); lp_b = rt.get_logprobs(c0, chain)
        check('G2b_logprob_repeatable', torch.equal(lp_a, lp_b), 'equal', 'compared')
        torch.manual_seed(6)
        s30 = rt(cond=c30, deterministic=False, return_chain=True, use_base_policy=False)
        own = rt.get_logprobs(c30, s30.chains).mean().item(); true = rt.get_logprobs(c0, s30.chains).mean().item()
        check('G3_rot_chain_less_likely_under_true', true < own, 'true < own', (true, own))
        full = norm.unnormalize(chain)
        m = torch.as_tensor(rx.mirror_full(full.cpu().numpy(), yaw), device=full.device, dtype=full.dtype)
        mn = norm.normalize(m)
        back = norm.unnormalize(mn)
        check('G4_mirror_chain_renormalize', torch.allclose(back, m, atol=1e-4), 'roundtrip', float((back - m).abs().max()))
        lpm = rt.get_logprobs(c0, mn)
        check('G4b_mirror_logprob_finite', bool(torch.isfinite(lpm).all()), 'finite', float(lpm.mean()))
    # G5 floor: deterministic output must not depend on the floor; random spread must grow
    outs = {}
    for floor in (.1, .2):
        with C.protocol(system, deterministic=True, min_std=floor):
            torch.manual_seed(9)
            with torch.no_grad():
                outs[floor] = rt(cond=c0, deterministic=True, return_chain=True, use_base_policy=False).trajectories
    check('G5_deterministic_floor_independent', torch.equal(outs[.1], outs[.2]), 'identical', float((outs[.1] - outs[.2]).abs().max()))
    spread = {}
    for floor in (.1, .2):
        with C.protocol(system, deterministic=False, min_std=floor):
            torch.manual_seed(11)
            with torch.no_grad():
                cond = {k: v.repeat(32, *([1] * (v.ndim - 1))) for k, v in c0.items()}
                tr = norm.unnormalize(rt(cond=cond, deterministic=False, return_chain=True, use_base_policy=False).trajectories)
        spread[floor] = float(tr[:, 5, :3].std(0).norm())
    check('G5b_random_spread_grows', spread[.2] > spread[.1], 'spread(0.2) > spread(0.1)', spread)
    # G6 RX clipped loss: zero advantages -> zero loss; one SGD step raises +A candidate vs -A candidate
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import AdvantageStats, compute_official_components
    with torch.no_grad():
        torch.manual_seed(21)
        cA = rt(cond=c0, deterministic=False, return_chain=True, use_base_policy=False).chains
        torch.manual_seed(22)
        cB = rt(cond=c30, deterministic=False, return_chain=True, use_base_policy=False).chains
        chains = torch.cat([cA, cB]); cond2 = {k: torch.cat([v, v]) for k, v in c0.items()}
        old = rt.get_logprobs(cond2, chains).reshape(2, rt.ft_denoising_steps, 16, 11)

    class B:
        pass

    def batch(adv):
        b = B(); k = torch.full((2,), rt.ft_denoising_steps - 1, dtype=torch.long, device=dev)
        rows = torch.arange(2)
        crit = dict(cond2)   # value part is ignored by RX; the critic is stubbed below
        b.condition, b.chains_prev, b.chains_next = crit, chains[rows, k], chains[rows, k + 1]
        b.denoising_inds, b.oldlogprobs = k, old[rows, k]
        b.advantages = torch.tensor(adv, device=dev); b.returns = torch.zeros(2, device=dev); b.oldvalues = torch.zeros(2, device=dev)
        return b
    one = torch.ones((), device=dev)
    stats = AdvantageStats(mean=0. * one, std=one - 1e-8, lower_quantile=-math.inf * one, upper_quantile=math.inf * one)
    params = [p for _, p in system.lora_bundle.named_adapter_parameters()]
    saved = [p.detach().clone() for p in params]
    object.__setattr__(rt, 'critic', lambda c: torch.zeros(2, 1, device=dev))
    try:
        p0 = compute_official_components(rt, batch([0., 0.]), stats, reward_start=2, reward_horizon=8).policy.mean()
        check('G6a_zero_adv_zero_loss', abs(float(p0)) < 1e-12, 0., float(p0))
        loss = compute_official_components(rt, batch([1., -1.]), stats, reward_start=2, reward_horizon=8).policy.mean()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    p -= 1e-3 * g / (sum(float((x ** 2).sum()) for x in grads if x is not None) ** .5 + 1e-12)
            new = rt.get_logprobs(cond2, chains).reshape(2, rt.ft_denoising_steps, 16, 11)
            d = (new - old)[:, -1, 2:10].mean(dim=(-1, -2))
        check('G6b_positive_adv_raised', float(d[0]) > float(d[1]), 'dlogp(+A) > dlogp(-A)', d.tolist())
    finally:
        del rt.__dict__['critic']
        with torch.no_grad():
            for p, s0 in zip(params, saved):
                p.copy_(s0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('cpu', 'gpu'))
    p.add_argument('--prepared')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    (cpu if args.mode == 'cpu' else gpu)(args)
    fails = sum(not r['passed'] for r in RESULTS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(mode=args.mode, results=RESULTS, fails=fails), indent=1))
    print('F12_PROBES', args.mode, 'total', len(RESULTS), 'fails', fails)
    sys.exit(fails)


if __name__ == '__main__':
    main()
