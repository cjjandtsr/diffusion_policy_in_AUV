#!/usr/bin/env python3
"""F10 T2 helper: per-denoising-step chain comparison, guidance off vs on.

Runs one real episode; at every decision the same encoded condition is sampled twice
with the same action seed (guidance off, then guidance on) and the two 11-step DDIM
chains are compared step by step. Nothing is trained and no committed trace is touched.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.routea_paired_support.core import file_hash, read_json, write_json
import scripts.run_routea_f10_eval as f10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='F10 run dir containing request.json')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--condition', type=int, default=0)
    parser.add_argument('--policy', default='P6', choices=sorted(f10.POLICY_BASE))
    parser.add_argument('--guidance', default='g1', choices=['g1', 'G0'])
    parser.add_argument('--eta', type=float)
    parser.add_argument('--decisions', type=int, default=20)
    parser.add_argument('--write', required=True)
    args = parser.parse_args()
    out, plan = f10.load_plan(args.output)
    system, config, header, overlays = f10.runtime(args, plan)
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy, reset_fixed_episode
    from scripts.routea_paired_support.runtime import make_adapter
    from scripts.run_routea_xk_evaluation import load_override
    from scripts.routea_denoising_guidance import Guidance, install
    from scripts.routea_paired_support.episodes import action_seed
    bank, tasks = f10.bank_maps(plan, args.split)
    entry = bank['episodes'][args.condition]
    task = tasks[entry['task_id']]
    world = bank['worlds'][task['world_id']]
    setting = dict(f10.GUIDANCE[args.guidance])
    if args.eta is not None:
        setting['eta'] = float(args.eta)
    policy = _E5Policy(system, use_base_policy=f10.POLICY_BASE[args.policy],
                       identity_hash=header['runtime_identity_hash'])
    adapter = make_adapter(system, config, world, task['role'], evaluation_protocol='E2')
    env = adapter.environment
    observation, context = reset_fixed_episode(env, task['payload'], seed=entry['reset_seed'],
                                              reset_callable=adapter.reset)
    scope = load_override(ROOT)(system.official_runtime, denoised_clip_value=2.,
                                min_std_vector=[.1] * 11, sampling_mode='deterministic')
    real = system.official_runtime
    results = []
    with scope:
        guide = Guidance(system.bc_runtime.policy.normalizer['action'].unnormalize,
                         setting['eta'], setting['consistency'], setting['d_safe'], setting['M'])
        restored_mean = real.p_mean_var
        identity = install(real, guide)
        guided_mean = real.p_mean_var
        for decision in range(args.decisions):
            encoded = system.observation_adapter.encode(
                observation, context['pointcloud_meta_history'],
                episode_identity_hash=context['episode_identity_hash'],
                decision_index=context['decision_index'])
            system.observation_adapter.verify(encoded)
            cond = encoded.condition
            seed = int(action_seed(entry['task_id'], entry['repeat'], decision))
            cloud = torch.as_tensor(observation['point_cloud'][-1], device=system.bc_runtime.device,
                                    dtype=torch.float32)
            guide.previous = None
            guide.begin(cloud, env._drone_pos, float(env._drone_yaw))
            real.p_mean_var = restored_mean
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base = real(cond=cond, deterministic=True, return_chain=True, use_base_policy=False)
            real.p_mean_var = guided_mean
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            guided = real(cond=cond, deterministic=True, return_chain=True, use_base_policy=False)
            equal = [bool(torch.equal(a, b)) for a, b in zip(base.chains[0], guided.chains[0])]
            first_difference = next((i for i, ok in enumerate(equal) if not ok), None)
            results.append(dict(decision=decision, action_seed=seed,
                                chain_steps=len(equal), chain_equal=equal,
                                first_differing_chain_index=first_difference,
                                triggered_steps=list(guide.triggered_steps),
                                cloud_points=int(guide.cloud.shape[0]),
                                guidance_identity=identity))
            bundle = policy.predict(observation, context, action_seed=seed)
            observation, reward, terminated, truncated, context = adapter.step(bundle)
            if terminated or truncated:
                break
    env.close()
    write_json(Path(args.write), dict(split=args.split, condition=args.condition,
                                      task_id=entry['task_id'], repeat=entry['repeat'],
                                      policy=args.policy, guidance=args.guidance, setting=setting,
                                      evaluation_sources=f10.source_identity(), rows=results))
    for row in results:
        print(json.dumps({k: row[k] for k in ('decision', 'first_differing_chain_index',
                                              'triggered_steps', 'cloud_points')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
