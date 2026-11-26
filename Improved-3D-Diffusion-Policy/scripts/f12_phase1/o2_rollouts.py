#!/usr/bin/env python3
"""O2 closed-loop rollouts: execute the first 4 steps of a candidate, then P6 #E2 to the end."""
import argparse
import copy
import csv
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.f12_phase1 import f12_common as C
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized


def clone_adapter(ad):
    new = copy.copy(ad)
    new.environment = copy.deepcopy(ad.environment)
    new.reward_state = copy.deepcopy(ad.reward_state)
    return new


def replay_to(ad, payload, reset_seed, prefix):
    obs, ctx = C.reset(ad, payload, reset_seed)
    for full in prefix:
        obs, rew, term, trunc, ctx = ad.step(ActionBundle.from_full(torch.as_tensor(full, dtype=torch.float32)))
        if term or trunc:
            raise RuntimeError('prefix terminated before branch state')
    return obs, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default=str(C.OUT / 's0' / 's0_states.npz'))
    ap.add_argument('--pool', default=str(C.OUT / 's0' / 's0_task_pool.json'))
    ap.add_argument('--traces', default=str(C.OUT / 's0' / 'traces'))
    ap.add_argument('--candidates', default=str(C.OUT / 'o2' / 'o2_candidate_arrays.npz'))
    ap.add_argument('--plan', default=str(C.OUT / 'o2' / 'o2_rollout_plan.csv'))
    ap.add_argument('--out', default=str(C.OUT / 'o2' / 'o2_rollouts.csv'))
    ap.add_argument('--limit-states', type=int, default=0)
    args = ap.parse_args()
    z = np.load(args.states, allow_pickle=False)
    cand = np.load(args.candidates, allow_pickle=False)
    actions = cand['action']; plans = cand['plan']
    pool = C.read_json(args.pool)
    tasks = {}
    for kind in ('blocked', 'passable'):
        for t in pool[kind]:
            tasks[t['task_id']] = t
    bank = C.bank_worlds()
    worlds = {w['world_id']: w for w in bank['worlds'].values()}
    picks = list(csv.DictReader(open(args.plan, encoding='utf-8-sig')))
    by_state = {}
    for p in picks:
        by_state.setdefault(int(p['state_rank']), []).append(p)
    if args.limit_states:
        by_state = dict(list(by_state.items())[:args.limit_states])
    system, config, header, _ = C.load_system()
    dev = system.bc_runtime.device
    spec = system.observation_adapter.spec
    rows = []
    for sr, group_picks in by_state.items():
        i = int(np.where(z['kind'].astype(str) == 'A')[0][sr - 1])
        task = tasks[str(z['task_id'][i])]
        world = worlds[str(z['world_id'][i])]
        # prefix up to decision k
        trace = Path(args.traces) / f"{task['task_id']}_{int(z['repeat'][i])}.jsonl.gz"
        prefix = []
        with gzip.open(trace, 'rt', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('type') == 'transition' and rec['decision'] < int(z['decision'][i]):
                    prefix.append(np.asarray(rec['full_action_16x11'], np.float32))
        reset_seed = next(int(r['reset_seed']) for r in task['repeats']
                          if int(r['repeat']) == int(z['repeat'][i]))
        ad0 = C.adapter_for(system, config, world, task['role'])
        try:
            obs, ctx = replay_to(ad0, task['payload'], reset_seed, prefix)
            base_env = copy.deepcopy(ad0.environment)
            base_reward_state = copy.deepcopy(ad0.reward_state)
        finally:
            ad0.environment.close()
        # run 8 candidates together, one batched P6 call per decision
        picks_here = group_picks
        envs, ads = [], []
        for p in picks_here:
            e = copy.deepcopy(base_env)
            a = copy.copy(ad0); a.environment = e
            a.reward_state = copy.deepcopy(base_reward_state)
            ads.append(a); envs.append(e)
        # branch candidate actions
        fulls = []
        for p in picks_here:
            # recover candidate index from the candidate CSV row order
            # candidate arrays index = (state_rank-1)*7*32 + group_index*32 + candidate
            gi = ['G-orig', 'G-rot30', 'G-rot60', 'G-mirror', 'G-sigma0.2', 'G-sigma0.3', 'G-early'].index(p['group'])
            idx = (sr - 1) * 7 * 32 + gi * 32 + int(p['candidate'])
            fulls.append(actions[idx])
        active = []
        paths = []
        for j, (a, e, full) in enumerate(zip(ads, envs, fulls)):
            obs2, rew, term, trunc, ctx2 = a.step(ActionBundle.from_full(torch.as_tensor(full, dtype=torch.float32)))
            paths.append([np.asarray(p, np.float32).copy() for p in np.asarray(ctx2.get('executed_path'), np.float32)])
            active.append(not (term or trunc))
        # continue with deterministic P6, batched over active envs
        for step in range(60):
            live = [j for j in range(len(ads)) if active[j]]
            if not live:
                break
            obs_list, ctx_list = [], []
            # environment observations are available through a cached last obs; re-create by env._get_obs_dict
            for j in live:
                o = ads[j].environment._get_obs_dict()
                c = {'pointcloud_meta_history': ads[j].environment._pc_meta_history,
                     'episode_identity_hash': ads[j].environment._episode_identity_hash,
                     'decision_index': ads[j].environment._observation_frame_index}
                obs_list.append(o); ctx_list.append(c)
            conds = []
            with torch.no_grad():
                for o, c in zip(obs_list, ctx_list):
                    a = torch.as_tensor(o['agent_pos'][-3:][None], dtype=torch.float32, device=dev)
                    pc = torch.as_tensor(o['point_cloud'][-3:, :, :3][None], dtype=torch.float32, device=dev)
                    cond, _ = encode_e5_condition_with_normalized(system.bc_runtime.policy,
                                                                   {'agent_pos': a, 'point_cloud': pc}, spec)
                    conds.append(cond)
                cond = {k: torch.cat([c[k] for c in conds], 0) for k in conds[0]}
                with C.protocol(system, deterministic=True, min_std=.1, clip=2.0):
                    samples = system.official_runtime(cond=cond, deterministic=True, return_chain=True,
                                                      use_base_policy=False)
                    full = system.bc_runtime.policy.normalizer['action'].unnormalize(samples.trajectories).cpu().numpy()
            for k, j in enumerate(live):
                obs2, rew, term, trunc, ctx2 = ads[j].step(ActionBundle.from_full(torch.as_tensor(full[k], dtype=torch.float32)))
                paths[j].extend(np.asarray(ctx2.get('executed_path'), np.float32)[1:])
                if term or trunc:
                    active[j] = False
        # outcomes
        for j, p in enumerate(picks_here):
            goal = np.asarray(task['payload']['goal_pos'], float)
            path = np.asarray(paths[j], float)
            mind = float(np.min(np.linalg.norm(path - goal, axis=1)))
            reason = str(ads[j].environment._done_reason or 'none')
            rows.append(dict(state_rank=sr, group=p['group'], reason=p['reason'], candidate=int(p['candidate']),
                             outcome=reason, reached_1m=bool(reason == 'reached' or mind <= 1.0),
                             min_goal_distance_m=mind, decisions=len(paths[j]),
                             world_id=world['world_id'], task_id=task['task_id'], repeat=int(z['repeat'][i]),
                             decision=int(z['decision'][i])))
        for e in envs:
            e.close()
        print('O2_ROLLOUT', sr, '/', max(by_state), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    C.write_json(str(args.out).replace('.csv', '.json'), dict(
        schema='f12_o2_rollouts_v1', rows=len(rows), protocol='candidate first H4 then P6 #E2 deterministic',
        outcome='raw done_reason plus reached_1m = (raw reached or closest approach <= 1 m)'))
    print('O2_ROLLOUTS_DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
