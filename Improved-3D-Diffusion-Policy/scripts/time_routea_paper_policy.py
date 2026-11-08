"""Paper B: synchronized E2 inference timings on a frozen real initial observation."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'source-root', 'checkpoint', 'bank', 'policy', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--expected-update', type=int, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    from scripts.run_routea_xk_evaluation import native_system, load_override
    system, config, header = native_system(args)
    import torch
    from scripts.routea_paired_support.core import read_json, observation_hash, canonical_hash, file_hash
    from scripts.routea_paired_support.runtime import make_adapter
    from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
    from scripts.routea_paired_support.episodes import action_seed
    bank = read_json(args.bank)
    if canonical_hash({k: v for k, v in bank.items() if k != 'content_hash'}) != bank['content_hash']:
        raise ValueError('bank content changed')
    entry = bank['episodes'][0]
    task = next(t for t in bank['tasks'] if t['task_id'] == entry['task_id'])
    adapter = make_adapter(system, config, bank['worlds'][task['world_id']], task['role'])
    try:
        observation, context = reset_fixed_episode(adapter.environment, task['payload'],
            seed=entry['reset_seed'], reset_callable=adapter.reset)
        if observation_hash(observation) != entry['observation_hash']:
            raise ValueError('real initial observation differs from bank')
        if system.bc_runtime.device.type != 'cuda':
            raise ValueError('paper timing requires CUDA on the execution machine')
        timings = []
        with load_override(args.repo)(system.official_runtime, denoised_clip_value=2.,
                min_std_vector=[.1] * 11, sampling_mode='deterministic') as protocol, torch.no_grad(), \
                torch.jit.optimized_execution(False):
            for index in range(220):
                seed = action_seed(entry['task_id'], entry['repeat'], 0)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()
                start = time.perf_counter()
                encoded = system.observation_adapter.encode(observation, context['pointcloud_meta_history'],
                    episode_identity_hash=context['episode_identity_hash'], decision_index=context['decision_index'])
                system.observation_adapter.verify(encoded)
                torch.cuda.synchronize()
                encoding = time.perf_counter()
                samples = system.official_runtime(cond=encoded.condition, deterministic=True,
                    return_chain=True, use_base_policy=args.policy == 'bc')
                torch.cuda.synchronize()
                denoising = time.perf_counter()
                full = system.bc_runtime.policy.normalizer['action'].unnormalize(samples.trajectories)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if tuple(full.shape[-2:]) != (16, 11):
                    raise ValueError('wrong action shape')
                if index >= 20:
                    timings.append([1000 * (encoding - start), 1000 * (denoising - encoding),
                                    1000 * (end - start)])
        values = np.asarray(timings)
        stats = {name: dict(mean_ms=float(values[:, i].mean()), median_ms=float(np.median(values[:, i])),
                           p95_ms=float(np.percentile(values[:, i], 95)))
                 for i, name in enumerate(('encode_and_verify', 'ddim', 'end_to_end'))}
        result = dict(policy=args.policy, actual_use_base_policy=args.policy == 'bc', warmup=20, repeats=200,
            stats=stats, samples_ms=timings, gpu=torch.cuda.get_device_name(system.bc_runtime.device),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            checkpoint_state_sha256=file_hash(Path(args.checkpoint) / 'state.pt'),
            runtime_identity=header['runtime_identity_hash'], bank_hash=bank['content_hash'],
            initial_hash=entry['initial_hash'], sampling_protocol=protocol,
            script_sha256=file_hash(__file__),
            critic_parameters=sum(p.numel() for p in system.critic.parameters()),
            critic_trainable_parameters=sum(p.numel() for p in system.critic.parameters() if p.requires_grad),
            note='Same frozen initial observation for all 200 measurements; seeds set before timing. '
                 'Encoding includes provenance verification; end-to-end includes action unnormalization.')
        with Path(args.output).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print('PAPER_TIMING_PASS', stats)
    finally:
        adapter.environment.close()


if __name__ == '__main__':
    import torch
    with torch.jit.optimized_execution(False):
        main()
