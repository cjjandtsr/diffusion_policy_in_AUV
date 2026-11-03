"""Focused checks for the common-protocol EGO adapter (no ROS required)."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.routea_paired_support.ego import world_cloud, reference_actions, sample_reference
from scripts.routea_paired_support.core import action_views
from scripts.run_routea_paired_comparison import parser


class EgoContractTests(unittest.TestCase):
    def test_default_map_size_survives_json_roundtrip(self):
        import json
        args = parser().parse_args([])
        self.assertEqual(args.ego_map_size,json.loads(json.dumps(args.ego_map_size)))

    def test_paper_source_receipt_preserves_bank(self):
        import copy
        from unittest.mock import patch
        from scripts.routea_paired_support.ego_workflow import recorded_sources
        bank = {'schema':'routea_paper_test_v1','spec_hash':'spec','content_hash':'frozen'}
        before = copy.deepcopy(bank)
        receipt = {'spec_hash':'spec','evaluation_sources':{'env.py':'hash'}}
        with patch('scripts.routea_paired_support.ego_workflow.read_json',return_value=receipt):
            self.assertEqual(recorded_sources(bank,Path('/bank/bank.json')),receipt['evaluation_sources'])
        self.assertEqual(bank,before)
        with patch('scripts.routea_paired_support.ego_workflow.read_json',return_value=dict(receipt,spec_hash='wrong')):
            with self.assertRaises(ValueError):
                recorded_sources(bank,Path('/bank/bank.json'))
        self.assertEqual(recorded_sources({'evaluation_sources':{'x':'y'}},Path('/bank/bank.json')),{'x':'y'})

    def test_native_yaw_wrap_does_not_apply_linear_lowpass(self):
        from scripts.routea_paired_support.ego import native_yaw_step
        yaw, rate = native_yaw_step(-3.13,3.13,0.,.01)
        self.assertAlmostEqual(yaw,-3.13)
        self.assertAlmostEqual(rate,math.pi/2)

    def test_cloud_body_to_world(self):
        result = world_cloud([[1,0,0],[0,1,0]], [3,4,5], math.pi/2)
        np.testing.assert_allclose(result, [[3,5,5],[2,4,5]], atol=1e-6)

    def test_reference_round_trip_and_branch_cut(self):
        from diffusion_policy_3d.env.traj_utils import action_to_world
        pos = np.array([3.,4.,2.], np.float32)
        ref = np.zeros((8,11), np.float32)
        ref[:,:3] = pos + np.arange(1,9)[:,None]*np.array([.1,-.03,.02])
        ref[:,3:6] = [.2,-.1,.05]
        ref[:,6:9] = [.01,.02,.03]
        ref[:,9] = -3.1
        bundle = reference_actions(ref, pos, 3.1)
        action_views(bundle.full[0],bundle.policy[0],bundle.rollout_executed[0])
        restored = action_to_world(bundle.policy[0],pos,np.zeros(3),3.1)
        np.testing.assert_allclose(restored,ref,atol=1e-6)
        self.assertLess(abs(bundle.policy[0,0,9]),.1)
        self.assertTrue(np.all(bundle.full[0,:2] == 0))
        self.assertTrue(np.all(bundle.full[0,10:] == 0))

    def test_bspline_linear_and_end_hold(self):
        # Clamped cubic Bezier x(t)=t on [0,1].
        msg = SimpleNamespace(order=3,knots=[0,0,0,0,1,1,1,1],
            pos_pts=[SimpleNamespace(x=x,y=0.,z=2.) for x in [0.,1/3,2/3,1.]],
            start_time=SimpleNamespace(to_sec=lambda:1.))
        result = sample_reference(msg,1.,np.array([0,0,2]),0.)
        np.testing.assert_allclose(result[:,0], [.3,.6,.9,1,1,1,1,1],atol=1e-6)
        np.testing.assert_allclose(result[:3,3],1.,atol=1e-6)
        np.testing.assert_array_equal(result[3:,3:9],0.)
        np.testing.assert_array_equal(result[:,9:],0.)

    def test_old_cli_and_new_cli(self):
        a = parser().parse_args(['--checkpoint','somewhere'])
        self.assertEqual(a.policy,'learned')
        self.assertEqual(a.checkpoint,['somewhere'])
        b = parser().parse_args(['--policy','ego','--task-bank','bank.json'])
        self.assertIsNone(b.checkpoint)
        self.assertEqual(b.policy,'ego')

    def test_intermediate_frames_do_not_trigger_new_execution_plan(self):
        from scripts.routea_paired_support.ego import EgoPolicy
        policy = object.__new__(EgoPolicy)
        policy.elapsed = 0.
        policy.calls = []
        policy.reference = np.zeros((8,11),np.float32)
        policy.env = SimpleNamespace(_drone_vel=np.zeros(3))
        policy.frames = [(np.array([i,0,2.]),0.,np.zeros((4096,3))) for i in range(4)]
        delivered = []
        def call(*args, **kwargs):
            delivered.append((args[4],kwargs['decision_boundary']))
        policy._call = call
        policy.after_step(dict(executed_time_s=1.2,done_reason='none'))
        np.testing.assert_allclose([t for t,_ in delivered],[.3,.6,.9,1.2])
        self.assertEqual([boundary for _,boundary in delivered],[False,False,False,True])
        policy.elapsed=0.
        policy.frames=policy.frames[:2]
        delivered.clear()
        policy.after_step(dict(executed_time_s=.45,done_reason='collision'))
        self.assertEqual(delivered,[(.3,False)])
        self.assertEqual(policy.elapsed,.45)

    def test_pair_rejects_changed_initial_or_policy(self):
        from scripts.routea_paired_support.ego_workflow import check_pair
        entry=dict(task_id='t',repeat=1,initial_hash='i',reset_seed=42)
        task=dict(payload={'x':1},world_id='w')
        bank=dict(content_hash='b',worlds={'w':dict(obstacle_sha256='o')})
        header=dict(task_id='t',repeat=1,task=task['payload'],bank_hash='b',
                    initial_hash='i',reset_seed=42,obstacle_sha256='o',
                    terminate_at_space_limits=False,policy='ego')
        summary=dict(initial_hash='i',bank_hash='b',policy='ego')
        check_pair(header,summary,entry,task,bank)
        with self.assertRaises(ValueError):
            check_pair(header,dict(summary,policy='bc'),entry,task,bank)
        with self.assertRaises(ValueError):
            check_pair(dict(header,initial_hash='wrong'),summary,entry,task,bank)


if __name__ == '__main__':
    unittest.main()
