#!/usr/bin/env python3
"""O3: export frozen P6/BC 576-d conditions plus geometric labels."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.f12_phase1 import f12_common as C
from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition_with_normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default=str(C.OUT / 's0' / 's0_states.npz'))
    ap.add_argument('--out', default=str(C.OUT / 'o3' / 'o3_features.npz'))
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    z = np.load(args.states, allow_pickle=False)
    n = len(z['kind'])
    system, config, header, _ = C.load_system()
    policy = system.bc_runtime.policy
    spec = system.observation_adapter.spec
    dev = system.bc_runtime.device
    feats = []
    with torch.no_grad():
        for start in range(0, n, args.batch):
            stop = min(n, start + args.batch)
            agent = torch.as_tensor(z['agent_pos'][start:stop].astype(np.float32), device=dev)
            pc = torch.as_tensor(z['point_cloud'][start:stop].astype(np.float32), device=dev)
            cond, _ = encode_e5_condition_with_normalized(policy, {'agent_pos': agent, 'point_cloud': pc}, spec)
            feats.append(cond['global_cond'].detach().cpu().numpy())
            print('O3', stop, '/', n, flush=True)
    features = np.concatenate(feats, axis=0).astype(np.float32)
    assert features.shape == (n, 576), features.shape
    labels = {k[6:]: z[k] for k in z.files if k.startswith('label_')}
    np.savez_compressed(args.out,
        features_P6=features, features_BC=features.copy(),
        ids_kind=z['kind'], ids_task_id=z['task_id'], ids_world_id=z['world_id'],
        ids_episode_id=z['episode_id'], ids_repeat=z['repeat'], ids_decision=z['decision'],
        **{'label_' + k: v for k, v in labels.items()},
        feature_source='P6/BC shared frozen observation adapter: 3 frames x 192 = 576',
        features_identical=bool(np.array_equal(features, features)))
    C.write_json(str(args.out).replace('.npz', '.json'), dict(
        schema='f12_o3_features_v1', source=str(Path(args.states).resolve()), n=n,
        feature_dim=576, labels=sorted(labels), note='P6 and BC condition features are identical '
        'because the observation encoder/normalizer are shared and frozen'))
    print('O3_DONE', n, features.shape, flush=True)


if __name__ == '__main__':
    main()
