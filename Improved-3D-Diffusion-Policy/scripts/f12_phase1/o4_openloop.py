#!/usr/bin/env python3
"""O4: open-loop BC/P6 amplitude on 3000 stratified BC-training frames."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.f12_phase1 import f12_common as C
from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized
from diffusion_policy_3d.common.sampler import get_val_mask


def sample_frames(ends, n, seed=20261126):
    val = get_val_mask(len(ends), .1, 42)
    train = [i for i in range(len(ends)) if not val[i]]
    starts = np.concatenate([[0], ends[:-1]])
    rng = np.random.default_rng(seed)
    per = n // len(train)
    extra = n - per * len(train)
    picked = []
    for j, ep in enumerate(train):
        lo, hi = int(starts[ep]) + 2, int(ends[ep]) - 17
        if hi < lo:
            continue
        k = per + (1 if j < extra else 0)
        k = min(k, hi - lo + 1)
        picked.extend(int(x) for x in rng.choice(np.arange(lo, hi + 1), k, replace=False))
    if len(picked) < n:
        all_valid = []
        for ep in train:
            lo, hi = int(starts[ep]) + 2, int(ends[ep]) - 17
            if hi >= lo:
                all_valid.extend(range(lo, hi + 1))
        rest = sorted(set(all_valid) - set(picked))
        picked.extend(int(x) for x in rng.choice(rest, n - len(picked), replace=False))
    return np.asarray(sorted(picked[:n]), np.int64)


def batch_predict(system, pol, agent, pc, base, batch=32):
    out = []
    dev = system.bc_runtime.device
    spec = system.observation_adapter.spec
    with torch.no_grad(), C.protocol(system, deterministic=True, min_std=.1, clip=2.0):
        for s in range(0, len(agent), batch):
            a = torch.as_tensor(agent[s:s + batch], dtype=torch.float32, device=dev)
            p = torch.as_tensor(pc[s:s + batch], dtype=torch.float32, device=dev)
            cond, _ = encode_e5_condition_with_normalized(system.bc_runtime.policy,
                                                           {'agent_pos': a, 'point_cloud': p}, spec)
            samples = system.official_runtime(cond=cond, deterministic=True, return_chain=True,
                                              use_base_policy=base)
            full = system.bc_runtime.policy.normalizer['action'].unnormalize(samples.trajectories)
            out.append(full.detach().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr')
    ap.add_argument('--out', default=str(C.OUT / 'o4' / 'o4_openloop.csv'))
    ap.add_argument('--n', type=int, default=3000)
    args = ap.parse_args()
    g = zarr.open(args.zarr, mode='r')
    state = g['data/state'][:]
    action = g['data/action'][:]
    ends = g['meta/episode_ends'][:]
    frames = sample_frames(ends, args.n)
    ep_of = np.searchsorted(ends, frames, side='right')
    agent = np.stack([state[frames - 2], state[frames - 1], state[frames]], axis=1)
    pc = np.stack([g['data/point_cloud'][f - 2:f + 1, :, :3] for f in frames], axis=0).astype(np.float32)
    static_agent = np.repeat(agent[:, -1:, :], 3, axis=1)
    static_pc = np.repeat(pc[:, -1:, :, :], 3, axis=1)
    labels = np.stack([action[f:f + 4, :3] for f in frames], 0)
    system, config, header, _ = C.load_system()
    print('predict BC', len(frames), flush=True)
    bc = batch_predict(system, None, agent, pc, True)
    print('predict P6', flush=True)
    p6 = batch_predict(system, None, agent, pc, False)
    print('predict BC static', flush=True)
    bcs = batch_predict(system, None, static_agent, static_pc, True)
    print('predict P6 static', flush=True)
    p6s = batch_predict(system, None, static_agent, static_pc, False)
    rows = []
    for j, f in enumerate(frames):
        row = dict(frame=int(f), episode=int(ep_of[j]), repeat=int(ep_of[j]),
                   label_len_1=float(np.linalg.norm(labels[j, 0])),
                   label_len_2=float(np.linalg.norm(labels[j, 1])),
                   label_len_3=float(np.linalg.norm(labels[j, 2])),
                   label_len_4=float(np.linalg.norm(labels[j, 3])),
                   label_first4_sum=float(sum(np.linalg.norm(labels[j, k]) for k in range(4))))
        for name, arr in (('BC', bc), ('P6', p6), ('BC_static', bcs), ('P6_static', p6s)):
            row[name + '_len_1'] = float(np.linalg.norm(arr[j, 0, :3]))
            row[name + '_len_2'] = float(np.linalg.norm(arr[j, 1, :3]))
            row[name + '_len_3'] = float(np.linalg.norm(arr[j, 2, :3]))
            row[name + '_len_4'] = float(np.linalg.norm(arr[j, 3, :3]))
            row[name + '_first4_sum'] = float(sum(np.linalg.norm(arr[j, k, :3]) for k in range(4)))
        rows.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    C.write_json(str(args.out).replace('.csv', '.json'), dict(
        schema='f12_o4_openloop_v1', frames=len(frames), source=args.zarr,
        split='BC train: val_ratio=0.1, seed=42', sample='episode-stratified frames; valid t in [start+2,end-17]',
        columns='displacement norms of 4-step actions in the unnormalized action frame'))
    print('O4_DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
