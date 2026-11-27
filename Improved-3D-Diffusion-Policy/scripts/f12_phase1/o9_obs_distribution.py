#!/usr/bin/env python3
"""O9a/b/c observation-distribution checks (phase 1b, Claude 2026-11-27). Inference only.

  python scripts/f12_phase1/o9_obs_distribution.py --selftest
  python scripts/f12_phase1/o9_obs_distribution.py --out <dir>

O9a  open-loop direction on demo frames: the 3000 O4 frames + 2000 extra stratified by the O7 front
     free distance so that >= 1000 frames have F in [0,3) m.  BC and P6 #E2 deterministic.
O9b  input swap on 300 (demo frame, S0 A/C state) pairs: DD / SD / DS / SS (point cloud / state),
     3-frame histories replaced whole.  BC deterministic.
O9c  point-cloud statistics of the 300 demo and 300 S0 current frames; clouds saved float16.
All predictions: torch seed fixed per batch (seed 20261127 + batch index), batch 32, fixed order.
"""
import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.f12_phase1.o7_demo_reaction import free_distance as o7_free  # noqa: E402

SEED = 20261127
Z_GROUND = .15


def yaw_of(state):
    return math.atan2(float(state[7]), float(state[8]))


def body_vec(world_delta, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    x, y, z = (float(v) for v in world_delta)
    return np.array([c * x - s * y, s * x + c * y, z])


def deviation_deg(vb, theta):
    if math.hypot(vb[0], vb[1]) < 1e-9:
        return float('nan')
    return abs((math.degrees(math.atan2(vb[1], vb[0]) - theta) + 180) % 360 - 180)


def cloud_stats(pc, prev):
    """pc, prev: (n,3) body-frame current / previous frame."""
    nz = pc[np.any(pc != 0, axis=1)]
    out = dict(nonzero=int(len(nz)))
    if len(nz) == 0:
        out.update(d_p10=float('nan'), d_p50=float('nan'), d_p90=float('nan'), frac_lt3=float('nan'),
                   z_p10=float('nan'), z_p50=float('nan'), z_p90=float('nan'), nn_prev_mean=float('nan'))
        out.update({'bin_%d' % b: 0 for b in range(11)})
        return out
    d = np.linalg.norm(nz, axis=1)
    out.update(d_p10=float(np.percentile(d, 10)), d_p50=float(np.percentile(d, 50)),
               d_p90=float(np.percentile(d, 90)), frac_lt3=float((d < 3).mean()),
               z_p10=float(np.percentile(nz[:, 2], 10)), z_p50=float(np.percentile(nz[:, 2], 50)),
               z_p90=float(np.percentile(nz[:, 2], 90)))
    h = np.minimum(np.floor(d).astype(int), 10)
    out.update({'bin_%d' % b: int((h == b).sum()) for b in range(11)})   # bin_10 = >= 10 m
    pv = prev[np.any(prev != 0, axis=1)]
    if len(pv) == 0:
        out['nn_prev_mean'] = float('nan')
    else:
        best = np.full(len(nz), np.inf)
        for s in range(0, len(pv), 1024):
            dd = ((nz[:, None, :] - pv[None, s:s + 1024, :]) ** 2).sum(-1)
            best = np.minimum(best, dd.min(1))
        out['nn_prev_mean'] = float(np.sqrt(best).mean())
    return out


def selftest():
    z = np.zeros((4096, 3), np.float32)
    st = cloud_stats(z, z)
    assert st['nonzero'] == 0, st
    pc = np.zeros((10, 3), np.float32); pc[0] = [1, 0, 0]; pc[1] = [0, 2, 0]
    st = cloud_stats(pc, pc)
    assert st['nonzero'] == 2 and st['bin_1'] == 1 and st['bin_2'] == 1 and st['nn_prev_mean'] == 0. and st['frac_lt3'] == 1.
    s = np.zeros(9); s[7], s[8] = 1., 0.   # yaw = +90 deg
    vb = body_vec([0, 1, 0], yaw_of(s))
    assert np.allclose(vb, [1, 0, 0], atol=1e-12), vb
    assert abs(deviation_deg(np.array([1., 1., 0.]), 0.) - 45.) < 1e-9
    print('O9_SELFTEST ok: all-zero cloud -> 0 nonzero; bins; yaw 90 deg world +y -> body +x; deviation 45 deg')


def main():
    import torch
    import zarr
    from scripts.f12_phase1 import f12_common as C
    from scripts.f12_phase1.o4_openloop import sample_frames
    from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr')
    ap.add_argument('--states', default=str(C.OUT / 's0' / 's0_states.npz'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--scan', type=int, default=30000, help='candidate frames scanned for the extra stratum')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    g = zarr.open(args.zarr, mode='r')
    S = g['data/state'][:]; A = g['data/action'][:]; ends = g['meta/episode_ends'][:]; PC = g['data/point_cloud']
    system, config, header, _ = C.load_system()
    dev = system.bc_runtime.device
    spec = system.observation_adapter.spec
    norm = system.bc_runtime.policy.normalizer['action']

    def predict(agent, pc, base, tag):
        res = []
        with torch.no_grad(), C.protocol(system, deterministic=True, min_std=.1, clip=2.0):
            for b, s0 in enumerate(range(0, len(agent), 32)):
                torch.manual_seed(SEED + b)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED + b)
                cond, _ = encode_e5_condition_with_normalized(
                    system.bc_runtime.policy,
                    {'agent_pos': torch.as_tensor(agent[s0:s0 + 32], dtype=torch.float32, device=dev),
                     'point_cloud': torch.as_tensor(pc[s0:s0 + 32], dtype=torch.float32, device=dev)}, spec)
                smp = system.official_runtime(cond=cond, deterministic=True, return_chain=True, use_base_policy=base)
                res.append(norm.unnormalize(smp.trajectories).cpu().numpy())
        print('O9_PREDICT', tag, len(agent), flush=True)
        return np.concatenate(res, 0)

    def frame_inputs(frames):
        agent = np.stack([S[frames - 2], S[frames - 1], S[frames]], axis=1).astype(np.float32)
        pc = np.stack([PC[f - 2:f + 1, :, :3] for f in frames], axis=0).astype(np.float32)
        return agent, pc

    def front(f, pc_cur):
        theta = math.atan2(S[f, 4], S[f, 3])
        return o7_free(pc_cur, theta, float(S[f, 2])), theta

    # ---------------- O9a
    base = sample_frames(ends, 3000)
    starts = np.concatenate([[0], ends[:-1]])
    from diffusion_policy_3d.common.sampler import get_val_mask
    val = get_val_mask(len(ends), .1, 42)
    valid = np.concatenate([np.arange(int(starts[e]) + 2, int(ends[e]) - 16) for e in range(len(ends))
                            if not val[e] and int(ends[e]) - 17 >= int(starts[e]) + 2])
    base_F = np.array([front(int(f), PC[int(f), :, :3])[0] for f in base])
    rng = np.random.default_rng(SEED)
    cand = rng.choice(np.setdiff1d(valid, base), min(args.scan, len(valid) - len(base)), replace=False)
    cand.sort()
    candF = np.empty(len(cand))
    for s0 in range(0, len(cand), 2000):
        chunk = cand[s0:s0 + 2000]
        for j, f in enumerate(chunk):
            candF[s0 + j] = front(int(f), PC[int(f), :, :3])[0]
    need_lt3 = max(0, 1000 - int((base_F < 3).sum()))
    extra = []
    lt3_bins = [(0, 1), (1, 2), (2, 3)]
    for lo, hi in lt3_bins:
        pool = cand[(candF >= lo) & (candF < hi)]
        k = min(len(pool), int(math.ceil(need_lt3 / 3.)))
        extra.extend(rng.choice(pool, k, replace=False).tolist())
    rest_bins = [(3, 5), (5, 8), (8, 99)]
    remaining = 2000 - len(extra)
    for b, (lo, hi) in enumerate(rest_bins):
        pool = np.setdiff1d(cand[(candF >= lo) & (candF < hi)], extra)
        k = min(len(pool), remaining // (3 - b))
        extra.extend(rng.choice(pool, k, replace=False).tolist())
        remaining -= k
    extra = np.asarray(sorted(extra), np.int64)
    frames = np.concatenate([base, extra])
    source = ['o4'] * len(base) + ['extra'] * len(extra)
    agent, pc = frame_inputs(frames)
    pred = {'BC': predict(agent, pc, True, 'O9a BC'), 'P6': predict(agent, pc, False, 'O9a P6')}
    rows = []
    for j, f in enumerate(frames):
        f = int(f)
        F, theta = front(f, pc[j, -1])
        yaw = yaw_of(S[f])
        row = dict(frame=f, source=source[j], free_front=F, goal_az_deg=math.degrees(theta),
                   goal_dist=float(S[f, 6]), yaw=yaw)
        vecs = dict(label=body_vec(A[f:f + 4, :3].sum(0), yaw),
                    BC=body_vec(pred['BC'][j, :4, :3].sum(0), yaw), P6=body_vec(pred['P6'][j, :4, :3].sum(0), yaw))
        for k, v in vecs.items():
            row.update({k + '_bx': v[0], k + '_by': v[1], k + '_bz': v[2], k + '_len': float(np.linalg.norm(v)),
                        k + '_dev_deg': deviation_deg(v, theta)})
        rows.append(row)
    with open(out / 'o9a_openloop_direction.csv', 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('O9A_DONE', len(rows), 'F<3:', int(sum(r['free_front'] < 3 for r in rows)), flush=True)

    # ---------------- O9b
    zs = np.load(args.states, allow_pickle=False)
    kinds = zs['kind'].astype(str)
    s_idx = np.where((kinds == 'A') | (kinds == 'C'))[0]
    prng = np.random.default_rng(SEED + 1)
    demo_pick = np.sort(prng.choice(base, 300, replace=False))
    s_pick = prng.choice(s_idx, 300, replace=False)
    dA, dP = frame_inputs(demo_pick)
    sA = zs['agent_pos'][s_pick][:, -3:].astype(np.float32)
    sP = zs['point_cloud'][s_pick][:, -3:, :, :3].astype(np.float32)
    combos = dict(DD=(dP, dA), SD=(sP, dA), DS=(dP, sA), SS=(sP, sA))
    preds = {k: predict(a, p, True, 'O9b ' + k) for k, (p, a) in combos.items()}
    dd_again = predict(dA, dP, True, 'O9b DD repeat')
    rows = []
    for j in range(300):
        f = int(demo_pick[j]); si = int(s_pick[j])
        for k, (p, a) in combos.items():
            st = a[j, -1]
            yaw = yaw_of(st)
            theta = math.atan2(float(st[4]), float(st[3]))
            v = body_vec(preds[k][j, :4, :3].sum(0), yaw)
            rows.append(dict(pair=j, demo_frame=f, s0_index=si, s0_kind=kinds[si], input=k,
                             len4=float(sum(np.linalg.norm(preds[k][j, t, :3]) for t in range(4))),
                             bx=v[0], by=v[1], bz=v[2], dev_deg=deviation_deg(v, theta),
                             goal_az_deg=math.degrees(theta)))
    with open(out / 'o9b_swap.csv', 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    o4_agent, o4_pc = frame_inputs(demo_pick)
    probe3 = dict(dd_inputs_equal_o4_construction=bool(np.array_equal(o4_agent, dA) and np.array_equal(o4_pc, dP)),
                  dd_repeat_identical=bool(np.array_equal(preds['DD'], dd_again)))
    print('O9B_DONE', len(rows), probe3, flush=True)

    # ---------------- O9c
    rows, clouds, labels = [], [], []
    for j in range(300):
        st = cloud_stats(dP[j, -1], dP[j, -2]); st.update(source='demo', ref=int(demo_pick[j])); rows.append(st)
        clouds.append(dP[j, -1]); labels.append('demo:%d' % int(demo_pick[j]))
    for j in range(300):
        st = cloud_stats(sP[j, -1], sP[j, -2]); st.update(source='S0', ref=int(s_pick[j])); rows.append(st)
        clouds.append(sP[j, -1]); labels.append('S0:%d' % int(s_pick[j]))
    with open(out / 'o9c_stats.csv', 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / 'o9c_clouds.npz', clouds=np.stack(clouds).astype(np.float16), labels=np.asarray(labels))
    probe4 = cloud_stats(np.zeros((4096, 3), np.float32), np.zeros((4096, 3), np.float32))['nonzero'] == 0
    C.write_json(out / 'o9_probes.json', dict(probe3=probe3, probe4_zero_cloud_nonzero_is_0=bool(probe4),
        o9a=dict(frames=len(frames), base=len(base), extra=int(len(extra)), lt3_base=int((base_F < 3).sum()),
                 lt3_total=int((base_F < 3).sum()) + int(sum(1 for f in extra if candF[np.searchsorted(cand, f)] < 3)),
                 scanned=int(len(cand)))))
    print('O9C_DONE', len(rows), 'probe4', probe4, flush=True)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        main()
