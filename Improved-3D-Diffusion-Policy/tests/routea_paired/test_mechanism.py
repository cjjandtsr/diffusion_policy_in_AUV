"""Risks specific to the approved crossed-state / goal response experiment."""
import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class MechanismContracts(unittest.TestCase):
    def _orchestration(self, fail_last):
        from scripts.routea_paired_support import mechanism as m
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from dataclasses import dataclass
        @dataclass
        class Provenance:
            policy_input_hash: str = 'input'
        names=['bc','rl_u128','rl_u256']
        cat=dict(content_hash='cat',common=dict(bc_file_sha256='bcbytes',bc_identity_hash='bcid'),
            policies={n:dict(use_base_policy=n=='bc',update=u,runtime_identity_hash='rlid',state_sha256=n)
                      for n,u in zip(names,[0,128,256])})
        bank=dict(content_hash='bank',continuation_draws=[0,1,2],max_h4_cycles=12,
                  states=[dict(id=str(i),group='g',header=dict(task_id='t'+str(i))) for i in range(18)])
        loaded={'name':'bc'}
        def continuation(name,sid,draw):
            absorbed=(name=='rl_u128' and sid=='0' and draw==0)
            row=dict(seed=m.paired_seed(sid,draw,1),initial_noise_hash='noise',
                     observation_provenance=dict(policy_input_hash='input'),pre_state={},
                     decision=1,pointcloud_metadata_before=[])
            return dict(draw=draw,outcome='reached' if absorbed else 'window_end',
                        rows=[{}] if absorbed else [{},row])
        def sample(sid,draw):
            return dict(full_action_16x11=np.zeros((16,11)).tolist(),metrics={},
                draw=draw,seed=m.paired_seed(sid,draw),policy_input_hash='input',
                initial_noise_hash='noise',observation_provenance=dict(policy_input_hash='input'))
        def read(path):
            if path.name=='protocol.json':return dict(bank_hash='bank')
            if path.name=='models.json':return cat
            name=path.parent.name;sid=path.stem
            return dict(state_id=sid,policy=name,actual_use_base_policy=name=='bc',
                policy_identity='bcid' if name=='bc' else 'rlid',
                samples=[sample(sid,i) for i in range(3)],
                continuations=[continuation(name,sid,i) for i in range(3)])
        calls=[];writes={}
        def flight(policy,snapshot,sid,draw,first,**kwargs):
            calls.append(dict(policy=policy.name,source=snapshot[0].source,first=first,
                              kwargs=kwargs,sid=sid,draw=draw))
            result=continuation(policy.name,sid,draw)
            if first is None:
                result=continuation('cross',sid,draw)
                result['rows']=result['rows'][1:]
            if fail_last and len(calls)==162:result['outcome']='collision'
            return result
        def replay(system,config,task_bank,state):
            return (SimpleNamespace(environment=SimpleNamespace(goal_w=np.ones(3)),
                                    sid=state['id'],source='initial'),{}, {})
        def fork(snapshot,sample):
            result=copy.deepcopy(snapshot);result[0].source=loaded['name']
            reason='reached' if loaded['name']=='rl_u128' and result[0].sid=='0' and sample['draw']==0 else None
            return result,reason
        def policy_factory(system,use_base_policy,identity_hash):
            return SimpleNamespace(use_base_policy=use_base_policy,last_provenance=Provenance(),
                                   name='bc' if use_base_policy else loaded['name'])
        with TemporaryDirectory() as directory, \
             patch.object(m,'read_bank',return_value=bank),patch.object(m,'read_json',side_effect=read), \
             patch.object(m,'file_hash',return_value='hash'), \
             patch.object(m,'write_json',side_effect=lambda path,value:writes.update({str(path):value})), \
             patch.object(m,'replay_state',side_effect=replay), \
             patch.object(m,'predict_with_noise',return_value=(None,'noise')), \
             patch.object(m,'action_record',side_effect=lambda *a:dict(full_action_16x11=np.zeros((16,11)).tolist(),metrics={})), \
             patch.object(m,'continue_flight',side_effect=flight), \
             patch.object(m,'fork_after_first',side_effect=fork), \
             patch.object(m,'snapshot_digest',return_value='snapshot'), \
             patch.object(m,'candidate_goals',return_value=[dict(valid=False)]) as goals, \
             patch('scripts.routea_paired_support.runtime.common_system',return_value=(None,None)), \
             patch('scripts.routea_paired_support.runtime.load_actor',side_effect=lambda r,s,d:loaded.update(name=d['state_sha256'])), \
             patch('scripts.routea_paired_support.workflow.policy_order',return_value=names), \
             patch('scripts.evaluate_dppo_uav_formal_endpoint._E5Policy',side_effect=policy_factory):
            if fail_last:
                with self.assertRaisesRegex(ValueError,'diagonal value'):
                    m.run_mechanism('.',cat,{},'bank',Path(directory)/'baseline',Path(directory)/'output')
            else:
                m.run_mechanism('.',cat,{},'bank',Path(directory)/'baseline',Path(directory)/'output')
        return calls,writes,goals.call_count

    def test_last_diagonal_failure_blocks_all_crossovers_and_goal_probes(self):
        calls,writes,goal_calls=self._orchestration(True)
        self.assertEqual(len(calls),162)
        self.assertEqual(goal_calls,0)
        self.assertTrue(any('diagonal_failure.json' in p for p in writes))
        self.assertFalse(any('diagonal_gate.json' in p or 'crossed' in p for p in writes))

    def test_crossed_directions_absorption_and_suffix_budget(self):
        calls,writes,goal_calls=self._orchestration(False)
        cross=[c for c in calls if c['first'] is None]
        self.assertEqual(len(cross),107)
        self.assertEqual(sum(c['source']=='rl_u256' and c['policy']=='rl_u128' for c in cross),54)
        self.assertEqual(sum(c['source']=='rl_u128' and c['policy']=='rl_u256' for c in cross),53)
        self.assertTrue(all(c['kwargs']==dict(maximum=11,cycle_start=1) for c in cross))
        self.assertFalse(any(c['source']=='rl_u128' and c['sid']=='0' and c['draw']==0 for c in cross))
        summary=next(value for path,value in writes.items() if path.endswith('summary.json'))
        self.assertEqual(summary['crossed_cases'],108)
        self.assertEqual(summary['absorbed_crossed'],1)
        self.assertEqual(goal_calls,18)

    def test_summary_does_not_treat_repeated_states_as_independent_tasks(self):
        from scripts.routea_paired_support.mechanism import summarize_cells
        rows=[]
        for sid,task,success in [('a','task1',True),('b','task1',True),('c','task2',False)]:
            for draw in range(3):
                rows.append(dict(state_id=sid,task_id=task,group='g',draw=draw,
                    outcomes={k:'reached' if success else 'window_end' for k in 'ABCD'}))
        result=summarize_cells(rows)
        self.assertEqual(result['task_count'],2)
        self.assertEqual(result['state_count'],3)
        self.assertEqual(result['task_equal_arrival_fraction']['A'],.5)

    def test_diagonal_rejects_changed_action_even_with_same_outcome(self):
        from scripts.routea_paired_support.mechanism import assert_diagonal
        old = dict(outcome='reached', rows=[dict(full_action_16x11=[[1.]], seed=9)])
        assert_diagonal(old, copy.deepcopy(old))
        changed = copy.deepcopy(old)
        changed['rows'][0]['full_action_16x11'][0][0] += 1e-7
        with self.assertRaisesRegex(ValueError, 'full_action'):
            assert_diagonal(old, changed)

    def test_retarget_updates_all_history_without_rotating_geometry(self):
        from scripts.routea_paired_support.mechanism import retarget_observation
        from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
        env = SimpleNamespace(_world_to_odom=lambda g: np.asarray(g)-[10, 0, 0],
                              _build_state=KinematicUAVEnv._build_state)
        states = np.array([KinematicUAVEnv._build_state([i, 0, 1], yaw, [4, 0, 1])
                           for i, yaw in [(0, 0), (1, np.pi/2), (2, np.pi)]])
        obs = dict(agent_pos=states.copy(), point_cloud=np.ones((3, 4, 3), np.float32))
        new = retarget_observation(env, obs, np.array([12, 3, 1]))
        np.testing.assert_array_equal(obs['agent_pos'], states)
        np.testing.assert_array_equal(new['point_cloud'], obs['point_cloud'])
        np.testing.assert_array_equal(new['agent_pos'][:, :3], states[:, :3])
        for row in new['agent_pos']:
            yaw = np.arctan2(row[7], row[8])
            world_delta = np.array([np.cos(yaw)*row[3]-np.sin(yaw)*row[4],
                                   np.sin(yaw)*row[3]+np.cos(yaw)*row[4], row[5]])
            np.testing.assert_allclose(row[:3]+world_delta, [2, 3, 1], atol=1e-6)

    def test_continuous_minimum_detects_ball_crossing_between_waypoints(self):
        from scripts.routea_paired_support.mechanism import goal_response_metrics
        path = np.array([[0, 0, 1], [2, 0, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]])
        m = goal_response_metrics(path, np.array([1, 0, 1]))
        self.assertEqual(m['h4_target_projection_m'], 4.)
        self.assertEqual(m['h4_min_goal_center_distance_m'], 0.)
        self.assertEqual(m['h4_min_goal_ball_distance_m'], 0.)

    def test_target_previously_crossed_in_visible_history_is_marked_invalid(self):
        from scripts.routea_paired_support.mechanism import candidate_goals
        env=SimpleNamespace(_drone_pos=np.array([2.,0.,1.]),goal_w=np.array([4.,0.,1.]),
            _drone_yaw=0.,_spawn_offset=np.zeros(3),_state_history=np.array([
                [-1.,0.,1.]+[0.]*6,[1.,0.,1.]+[0.]*6,[2.,0.,1.]+[0.]*6]),
            obstacles=[],collision_thresh=.1,
            _curriculum_xy_range=lambda:(-14.,14.,-14.,14.),_curriculum_z_range=lambda:(1.,5.))
        targets=candidate_goals(env)
        rear=next(t for t in targets if t['label']=='bearing_+180')
        self.assertFalse(rear['valid'])
        self.assertIn('arrival_in_observed_history',rear['rejection_reasons'])
        self.assertTrue(next(t for t in targets if t['label']=='bearing_+0')['valid'])

    def test_remaining_window_and_absolute_cycle_seed(self):
        from scripts.routea_paired_support.recovery import continue_flight, paired_seed
        from unittest.mock import patch
        from dataclasses import dataclass
        @dataclass
        class Provenance:
            policy_input_hash: str = 'test'
        class Adapter:
            def __init__(self):
                self.environment=SimpleNamespace(episode_step=59,max_episode_steps=60,
                    goal_w=np.array([5.,0.,1.]),_drone_pos=np.array([0.,0.,1.]),
                    _drone_vel=np.zeros(3),_drone_yaw=0.)
            def step(self, bundle):
                self.environment.episode_step += 1
                return {}, 0., False, True, dict(done_reason='time_limit',
                    executed_path=[[0,0,1]],executed_time_s=1.2,pointcloud_meta_history=[])
        policy=SimpleNamespace(last_provenance=Provenance())
        with patch('scripts.routea_paired_support.recovery.action_record', return_value={}), \
             patch('scripts.routea_paired_support.recovery.predict_with_noise',
                   return_value=(SimpleNamespace(), 'noise')) as predict:
            result=continue_flight(policy,(Adapter(),{},dict(pointcloud_meta_history=[])),
                                   's',2,None,maximum=11,cycle_start=1)
        self.assertEqual(len(result['rows']),1)
        self.assertEqual(result['rows'][0]['cycle'],1)
        self.assertEqual(predict.call_args.args[-1],paired_seed('s',2,1))
        self.assertEqual(result['remaining_decisions'],0)

    def test_fork_keeps_terminated_prefix_absorbed(self):
        from scripts.routea_paired_support.mechanism import fork_after_first
        class Adapter:
            def __init__(self):
                self.environment=SimpleNamespace(episode_step=4)
            def step(self,bundle):
                self.environment.episode_step+=1
                return {},100.,True,False,dict(done_reason='reached')
        original=(Adapter(),{}, {})
        snapshot,reason=fork_after_first(original,dict(full_action_16x11=np.zeros((16,11))))
        self.assertEqual(reason,'reached')
        self.assertEqual(snapshot[0].environment.episode_step,5)
        self.assertEqual(original[0].environment.episode_step,4)


if __name__ == '__main__':
    unittest.main()
