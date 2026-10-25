"""Frozen-state diagnostics must not change the task or evaluation denominator."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.routea_paired_support.recovery import (
    paired_seed, check_replayed_state, clone_snapshot, continuation_limit,
    verify_pairing, recovery_counts)


class RecoveryContracts(unittest.TestCase):
    def test_reference_accepts_only_pinned_common_protocol_sources(self):
        from scripts.routea_paired_support.reference import COMMON_PROTOCOL_SOURCES, source_compatible
        for name, approved in COMMON_PROTOCOL_SOURCES.items():
            self.assertTrue(source_compatible(name, approved, 'old', True))
            self.assertFalse(source_compatible(name, 'unreviewed', 'old', True))
            self.assertFalse(source_compatible(name, approved, 'old', False))
        self.assertTrue(source_compatible('actor.py', 'same', 'same', True))
        self.assertFalse(source_compatible('actor.py', 'new', 'old', True))

    def test_shared_seed_distinct_draws_and_cycles(self):
        self.assertEqual(paired_seed('s', 3, 0), paired_seed('s', 3, 0))
        self.assertEqual(len({paired_seed('s', i, k) for i in range(50) for k in range(12)}), 600)

    def test_snapshot_clone_keeps_budget_and_independent_history(self):
        adapter = SimpleNamespace(environment=SimpleNamespace(episode_step=56,
            max_episode_steps=60, history=np.array([1., 2.]), rng=np.random.default_rng(4)))
        snapshot = (adapter, {'point_cloud': np.zeros((3, 2, 3))}, {'metadata': [1]})
        a, b = clone_snapshot(snapshot), clone_snapshot(snapshot)
        a[0].environment.history[0] = 99
        a[1]['point_cloud'][0, 0, 0] = 99
        a[2]['metadata'].append(2)
        self.assertEqual(b[0].environment.history[0], 1)
        self.assertEqual(b[1]['point_cloud'][0, 0, 0], 0)
        self.assertEqual(b[2]['metadata'], [1])
        self.assertEqual(a[0].environment.rng.random(), b[0].environment.rng.random())
        self.assertEqual(continuation_limit(b[0], 12), 4)

    def test_replay_rejects_history_or_position_mismatch(self):
        expected = {'position': [1, 2, 3], 'velocity': [0, 0, 0], 'yaw': .2}
        env = SimpleNamespace(_drone_pos=np.array([1, 2, 3]), _drone_vel=np.zeros(3), _drone_yaw=.2)
        provenance = {'policy_input_hash': 'x', 'normalized_history_hash': 'n', 'raw_history_hash': 'r'}
        check_replayed_state(env, expected, provenance, provenance)
        with self.assertRaises(ValueError):
            check_replayed_state(env, expected, {**provenance, 'raw_history_hash': 'wrong'}, provenance)
        env._drone_pos[0] = 8
        with self.assertRaises(ValueError):
            check_replayed_state(env, expected, provenance, provenance)

    def test_pairing_requires_actual_noise_and_input_identity(self):
        row = {'draw': 0, 'seed': 3, 'policy_input_hash': 'x', 'initial_noise_hash': 'n'}
        verify_pairing([row], [copy.deepcopy(row)])
        with self.assertRaises(ValueError):
            verify_pairing([row], [{**row, 'initial_noise_hash': 'different'}])

    def test_noise_hook_reads_actual_chain_and_is_removed(self):
        import torch
        from scripts.routea_paired_support.recovery import predict_with_noise
        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import canonical_array_hash
        class Runtime(torch.nn.Module):
            def forward(self, value):
                return SimpleNamespace(chains=value)
        runtime=Runtime()
        chain=torch.arange(12,dtype=torch.float32).reshape(1,2,2,3)
        def predict(obs,ctx,action_seed):
            runtime(chain)
            return 'action'
        policy=SimpleNamespace(system=SimpleNamespace(official_runtime=runtime),predict=predict)
        action,digest=predict_with_noise(policy,None,None,7)
        self.assertEqual(action,'action')
        self.assertEqual(digest,canonical_array_hash(chain[:,0]))
        self.assertEqual(len(runtime._forward_hooks),0)

    def test_diagnostic_counts_do_not_include_full_episode_scores(self):
        self.assertEqual(recovery_counts(24, 9, 50, 5),
                         {'states': 24, 'policies': 9, 'open_loop_samples': 10800,
                          'continuations': 1080, 'max_continuation_decisions': 12960})

    def test_a1_bank_is_outcome_stratified_and_preserves_original_clock(self):
        from collections import Counter
        from scripts.routea_paired_support.recovery import read_bank
        root=Path(__file__).resolve().parents[2]
        bank=read_bank(root/'scripts/routea_paired_support/a1_mechanism_bank_v1.json')
        self.assertEqual(Counter(s['group'] for s in bank['states']),
                         {'u256_near_goal_timeout':6,'u128_success_control':6,
                          'bc_success_control':6})
        self.assertTrue(all(s['decision']<=48 and len(s['prefix_actions'])==s['decision']
                            for s in bank['states']))
        self.assertEqual(recovery_counts(18,3,bank['draws'],len(bank['continuation_draws']),
                                         maximum=bank['max_h4_cycles']),
                         {'states':18,'policies':3,'open_loop_samples':1080,
                          'continuations':162,'max_continuation_decisions':1944})

    def test_continuation_keeps_actual_input_and_noise_after_replanning(self):
        from dataclasses import dataclass
        from unittest.mock import patch
        from scripts.routea_paired_support.recovery import continue_flight
        @dataclass
        class Provenance:
            policy_input_hash: str = 'next-input'
        class Adapter:
            environment=SimpleNamespace(episode_step=58,max_episode_steps=60,
                goal_w=np.array([5.,0.,1.]),_drone_pos=np.array([0.,0.,1.]),
                _drone_vel=np.zeros(3),_drone_yaw=0.)
            def step(self,bundle):
                self.environment.episode_step+=1
                return {},0.,False,self.environment.episode_step==60,dict(
                    done_reason='time_limit' if self.environment.episode_step==60 else 'none',
                    executed_path=[[0,0,1]],executed_time_s=1.2,pointcloud_meta_history=[])
        first=dict(full_action_16x11=np.zeros((16,11)),initial_noise_hash='first-noise',
                   observation_provenance={'policy_input_hash':'first-input'})
        policy=SimpleNamespace(last_provenance=Provenance())
        with patch('scripts.routea_paired_support.recovery.action_record',side_effect=lambda *args:{}), \
             patch('scripts.routea_paired_support.recovery.predict_with_noise',
                   return_value=(SimpleNamespace(), 'next-noise')) as sample:
            result=continue_flight(policy,(Adapter(),{}, {'pointcloud_meta_history':[]}), 's',0,first)
        self.assertEqual(sample.call_count,1)
        a,b=result['rows']
        self.assertEqual(a['initial_noise_hash'],'first-noise')
        self.assertEqual(b['initial_noise_hash'],'next-noise')
        self.assertEqual(b['observation_provenance']['policy_input_hash'],'next-input')
        self.assertEqual(a['source_sample_draw'],0)
        self.assertIn('pointcloud_metadata_before',b)


if __name__ == '__main__':
    unittest.main()
