"""F9 v1.3 A-level probes; no model, training, or WSL runs."""
import unittest
import ast
from contextlib import contextmanager
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import patch
import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import (
    OnlineKeep, eligible_indices, execution_delta, stage_gate, rng_scope, capture_rng, paired_sample,demand_counts)


def trainer_fixture():
    """Reuse the existing logical-update fixture, without its pytest-only runner."""
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import DDPMTrainer
    path=Path(__file__).parents[1]/'dppo_uav_rebuild/engine/test_logical_accumulation_v2.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in
           ('Bundle','Runtime','make_trainer','batch')]
    ns=dict(torch=torch,nn=torch.nn,SimpleNamespace=SimpleNamespace,DDPMTrainer=DDPMTrainer)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns


class ToySampler:
    def __init__(self):
        path=Path(__file__).parents[2]/'third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='p_mean_var')
        ns=dict(torch=torch,extract=lambda a,i,shape:a[i].reshape(-1,1,1))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        self.p_mean_var=MethodType(ns['p_mean_var'],self)
        self.use_ddim=True;self.predict_epsilon=False;self.ddim_steps=10;self.ft_denoising_steps=10
        self.recalculate_epsilon_after_clip=False;self.denoised_clip_value=2.;self.eps_clip_value=None
        self.ddim_alphas=torch.full((10,),.5);self.ddim_alphas_prev=torch.full((10,),.8)
        self.ddim_alphas_prev[-1]=1.
        self.ddim_sqrt_one_minus_alphas=torch.full((10,),.5**.5)
        self.actor=lambda x,t,cond:torch.zeros_like(x)
        self.actor_ft=self.actor
        self.eta=lambda cond:torch.ones(1,11)

    def __call__(self,cond,deterministic,return_chain,use_base_policy):
        x=torch.randn(1,16,11)
        for i in range(10):
            mean,logvar,_=self.p_mean_var(x,torch.tensor([i]),cond,index=torch.tensor([i]),deterministic=deterministic)
            noise=torch.randn_like(x).clamp(-10,10)
            std=0 if deterministic else torch.exp(.5*logvar).clamp_min(.1)
            x=mean+std*noise
        return SimpleNamespace(trajectories=x)


@contextmanager
def toy_scope(runtime,**kwargs):
    yield {}


class F9Probes(unittest.TestCase):
    def test_A7_parent_task_strata_gained_lost(self):
        from scripts.run_routea_f9 import task_strata
        def result(solved):
            return dict(rows=[dict(task_id=str(i),repeat=r,outcome='reached' if i in solved else 'collision')
                              for i in range(85) for r in range(1,6)])
        parent=set(range(49));control=parent|{50};treatment=(parent-{0})|{49}
        report=task_strata(result(treatment),result(control),result(parent))
        self.assertEqual(report['P6_solved_49']['T_vs_P6']['lost'],['0'])
        self.assertEqual(report['P6_unsolved_36']['T_vs_P6']['gained'],['49'])
        self.assertEqual(report['P6_unsolved_36']['T_vs_C'],dict(gained=['49'],lost=['50']))

    def test_A2_clamp_trigger_without_effective_correction(self):
        from scripts.routea_denoising_guidance import Guidance
        guide=Guidance(lambda x:x,.2,0)
        guide.begin(torch.empty(0,3),[0.,0.,-100.],0.)
        x=torch.zeros(1,16,11);x[:,:,2]=2.
        y=guide.update(x,torch.tensor([7]))
        self.assertTrue(guide.triggered_steps)
        self.assertTrue(torch.equal(x,y))
        self.assertEqual(execution_delta(x[0],y[0],[0,0,1],[0,0,0],0),0.)

    def test_A2_first_entry_effective_holdout(self):
        paths = [[[0, 0, 1], [1, 0, 1]], [[1, 0, 1], [2.2, 0, 1]], [[2.2, 0, 1], [3, 0, 1]]]
        # Goal x3: first strict <1 is decision1; only decision0 can enter.
        self.assertEqual(eligible_indices(paths, [3, 0, 1], [.06, .1, .2]), [0])
        self.assertEqual(eligible_indices(paths, [3, 0, 1], [.049, .1, .2]), [])
        self.assertEqual(eligible_indices(paths[:1], [3, 0, 1], [.2]), [])

    def test_A3_execution_delta_and_tail(self):
        a = np.zeros((16, 11), np.float32)
        b = a.copy(); b[10, 0] = 100.
        self.assertEqual(execution_delta(a, b, np.zeros(3), np.zeros(3), 0.), 0.)
        b[2:6, 0] = .02
        self.assertAlmostEqual(execution_delta(a, b, np.zeros(3), np.zeros(3), 0.), .08, places=6)

    def test_A4_four_mutually_exclusive_branches(self):
        self.assertEqual(stage_gate(.07, .1, 405, 425, .9), 'pass')
        self.assertEqual(stage_gate(.07, .1, 404, 425, .2), 'behavior_changed_closed_loop_worse')
        self.assertEqual(stage_gate(.08, .1, 425, 425, .6), 'restart')
        self.assertEqual(stage_gate(.08, .1, 425, 425, .5), 'fitted_without_student_change')
        self.assertEqual(stage_gate(.08, .1, 425, 425, .6, restarted=True), 'stop_after_restart')

    def test_A1_global_rng_isolation(self):
        torch.manual_seed(123)
        before = capture_rng()
        expected = torch.rand(5)
        torch.set_rng_state(before['cpu'])
        with rng_scope(seed=7):
            torch.rand(100)
            np.random.rand(100)
        self.assertTrue(torch.equal(expected, torch.rand(5)))

    def test_A2_fifo_holdout_and_labels(self):
        keep=OnlineKeep(1,capacity=2)
        paths=[[[i,0,1],[i+1,0,1]] for i in range(4)]
        records=[dict(sample_id=str(i),action=torch.zeros(16,11),condition=torch.zeros(576)) for i in range(4)]
        keep.ingest(records,paths,[4,0,1],[.1,.1,.1,.1],0)
        self.assertEqual(len(keep.records),0)
        self.assertEqual([r['intervention'] for r in keep.heldout],['first_intervention','after_intervention','after_intervention'])
        keep.ingest(records,paths,[4,0,1],[.1,.1,.1,.1],1)
        self.assertEqual([r['sample_id'] for r in keep.records],['1','2'])
        self.assertTrue(all(r['episode_number']%10 for r in keep.records))
        self.assertEqual(eligible_indices(paths,[4,0,1],[.1]*4,['collision','none','none','reached']),[])

    def test_A3_same_x0_core_and_sampler_state(self):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.selective_keep import prefix_x0_loss,executed_weights
        runtime=SimpleNamespace(predict_epsilon=False,ft_denoising_steps=10,ddim_t=torch.arange(10),
            ddim_alphas=torch.linspace(.1,.9,10),actor_ft=lambda x,t,c:torch.zeros_like(x))
        keep=OnlineKeep(11,batch_size=2)
        keep.records=[dict(action=torch.ones(16,11),condition=torch.zeros(576))]
        saved=keep.state_dict()
        self.assertEqual(float(keep.loss(runtime,device='cpu')),1.) # 16 weighted entries /16.
        keep.load_state_dict(saved)
        self.assertTrue(torch.equal(keep.generator.get_state(),saved['rng']))

    def test_A4_exact_four_batch_calibration(self):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.selective_keep import calibrate_keep
        fixture=trainer_fixture();trainer,runtime=fixture['make_trainer']()
        runtime.predict_epsilon=False;runtime.ddim_t=torch.arange(10);runtime.ddim_alphas=torch.linspace(.1,.9,10)
        runtime.actor_ft=lambda x,t,c:runtime.actor_parameter.expand_as(x)
        keep=OnlineKeep(3,batch_size=2)
        keep.records=[dict(action=torch.ones(16,11),condition=torch.zeros(576))]
        before=keep.generator.get_state().clone()
        result=calibrate_keep(trainer,[fixture['batch'](2) for _ in range(4)],keep,target_ratio=1.)
        expected=np.median([r['ppo_norm'] for r in result['batches']])/np.median([r['keep_norm'] for r in result['batches']])
        self.assertEqual(result['coefficient'],expected)
        self.assertTrue(torch.equal(before,keep.generator.get_state()))
        print('SYNTHETIC_F9_CALIBRATION',json.dumps(result))

    def test_A5_random_chain_identical_noise_and_restore(self):
        runtime=ToySampler();original=runtime.p_mean_var
        torch.manual_seed(24)
        with patch('scripts.run_routea_xk_evaluation.load_override',return_value=toy_scope):
            pair=paired_sample(runtime,{'state':torch.zeros(1,576)},lambda x:x,
                torch.tensor([[10.,10.,0.]]),[0.,0.,1.],[0.,0.,0.],0.,
                dict(eta=.2,d_safe=.6,M=2),deterministic=False,audit_noise=True)
        self.assertTrue(torch.equal(pair['full'],pair['base_full']))
        self.assertEqual(len(pair['noise'][0]),11)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(*pair['noise'])))
        self.assertIs(runtime.p_mean_var,original)
        self.assertEqual(pair['delta_exec'],0.)

    def test_A3_student_demand_denominator(self):
        paths=[[[0,0,1],[1,0,1]],[[1,0,1],[2.2,0,1]],[[2.2,0,1],[3,0,1]]]
        self.assertEqual(demand_counts(paths,[3,0,1],[.05,.2,.3]),(1,1,1))
        self.assertEqual(demand_counts(paths[:1],[3,0,1],[.049]),(0,1,None))

    def test_A5_nonzero_guidance_same_noise(self):
        runtime=ToySampler();torch.manual_seed(24)
        with patch('scripts.run_routea_xk_evaluation.load_override',return_value=toy_scope):
            pair=paired_sample(runtime,{'state':torch.zeros(1,576)},lambda x:x,
                torch.tensor([[.2,0.,0.]]),[0.,0.,1.],[0.,0.,0.],0.,
                dict(eta=.4,d_safe=1.2,M=4),deterministic=False,audit_noise=True)
        self.assertTrue(pair['triggered'])
        self.assertGreater(pair['delta_exec'],.05)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(*pair['noise'])))

    def test_A5_two_layer_cosine_and_clip_actual_commit(self):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import TrainerConfig
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import AdvantageStats,compute_official_components
        fixture=trainer_fixture();trainer,runtime=fixture['make_trainer']()
        trainer.config=TrainerConfig(max_grad_norm=.001)
        trainer.record_f9=True
        p=runtime.actor_parameter
        trainer.keep=SimpleNamespace(coefficient=.3,generator=torch.Generator().manual_seed(2),
                                     loss=lambda runtime,device:(runtime.actor_parameter-1).square())
        trainer.f8_keep=SimpleNamespace(coefficient=.6,generator=torch.Generator().manual_seed(3),
                                        loss=lambda runtime,device:(runtime.actor_parameter+1).square())
        batch=fixture['batch'](8)
        parts=compute_official_components(runtime,batch,AdvantageStats.from_tensor(batch.advantages,.05,.95),
                                           reward_start=2,reward_horizon=4)
        gp,=torch.autograd.grad(parts.policy.mean(),[p])
        expected=abs(float(gp)+.3*2*(float(p)-1)+.6*2*(float(p)+1))
        metrics=trainer.update_logical_batch(batch,microbatch_pairs=8)
        d=metrics.f8_keep_diagnostics
        self.assertAlmostEqual(d['layers_cosine'],-1.,places=6)
        self.assertAlmostEqual(d['total_gradient_before_clip'],expected,places=6)
        self.assertAlmostEqual(d['total_gradient_after_clip'],.001,places=7)
        self.assertAlmostEqual(d['clip_multiplier'],.001/expected,places=6)
        self.assertEqual(trainer.counters.committed_updates,1)


def local_smoke(output):
    """Real H4 environment -> online buffer -> shared loss -> saved/restored sidecar.

    Synthetic actor and observations, NOT a P6 model or a training run.
    """
    import hashlib
    from dataclasses import dataclass
    from test_e3g_probe import environment
    from scripts.routea_paired_support.core import write_json,write_npz
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter
    from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v2.facade import RewardV2Facade
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f9_support import F9Session
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.f9_online import validate_teacher_trace
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    keep=OnlineKeep(11,batch_size=2)
    full=torch.zeros(1,16,11);full[:,:,0]=.25
    base=full.clone();base[:,:,0]=0.
    for episode in (0,1):
        env,_=environment('E2');env.record_execution_candidate=True
        env.goal_w=np.array([3.,0.,1.],np.float32)
        env._reward_history=dict(initial_distance=3.,best_distance=3.)
        facade=RewardV2Facade();adapter=KinematicAdapter(env,facade,facade.initial_state())
        records=[];paths=[];deltas=[];trace=[]
        header=dict(initial_position=env._drone_pos.tolist(),geometry=[],goal=env.goal_w.tolist(),
                    z_range=[0.,10.],terminate_at_space_limits=True)
        for decision in range(3):
            pre=env._drone_pos.copy()
            obs,reward,terminated,truncated,info=adapter.step(ActionBundle.from_full(full))
            records.append(dict(sample_id='e%d_d%d'%(episode,decision),task_id='SYNTHETIC',decision=decision,
                action=full[0].clone(),condition=torch.zeros(576),done_reason=info['done_reason']))
            paths.append(info['executed_path']);deltas.append(execution_delta(base[0],full[0],pre,[0,0,0],0.))
            trace.append(dict(decision=decision,pre_state=dict(position=pre.tolist(),velocity=[0.,0.,0.],yaw=0.),
                full_action=full[0].tolist(),base_full_action=base[0].tolist(),
                executed_path=info['executed_path'].tolist(),execution_candidate_path=info['execution_candidate_path'].tolist(),
                done_reason=info['done_reason'],executed_time_s=info['executed_time_s'],
                event_segment=info['event_segment'],event_fraction=info['event_fraction'],
                delta_exec=deltas[-1],effective=deltas[-1]>=.05))
            if terminated or truncated:break
        accepted=keep.ingest(records,paths,env.goal_w,deltas,episode)
        summary=dict(decisions=len(trace),done_reason=trace[-1]['done_reason'],eligible=len(accepted))
        self_check=validate_teacher_trace(header,trace,summary)
        assert self_check==[0,1]
        write_json(output/('episode%d.json'%episode),dict(header=header,rows=trace,summary=summary))
        for record in accepted:
            path=output/(record['sample_id']+'.npz')
            write_npz(path,dict(point_cloud=np.ones((3,4096,3),np.float32),agent_pos=np.zeros((3,9),np.float32)))
            record['observation_path']=str(path.resolve())
            record['observation_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        if episode==0:keep.heldout=accepted
        else:keep.records=accepted
    assert len(keep.records)==len(keep.heldout)==2
    runtime=SimpleNamespace(predict_epsilon=False,ft_denoising_steps=10,ddim_t=torch.arange(10),
        ddim_alphas=torch.linspace(.1,.9,10),actor_ft=lambda x,t,c:torch.zeros_like(x))
    loss=float(keep.loss(runtime,device='cpu'))
    session=F9Session.__new__(F9Session);session.directory=output;session.keep=keep
    session.sampler=SimpleNamespace(state_dict=lambda:dict(synthetic=True),load_state_dict=lambda x:None)
    session.rng=capture_rng();session.ordinal=2;session.total_transitions=6;session.prefill_transitions=0
    session.snapshot(256);state=session.state_dict()
    torch.save(state,output/'sidecar_state.pt')
    keep.records=[];keep.heldout=[]
    session.load_state_dict(torch.load(output/'sidecar_state.pt',weights_only=True))
    assert len(keep.records)==2 and len(keep.heldout)==2
    write_json(output/'result.json',dict(passed=True,synthetic_actor=True,native_model_loaded=False,
        actual_environment=True,training_records=len(keep.records),heldout_records=len(keep.heldout),
        shared_loss=loss,snapshot_sha256=session.snapshot_sha))
    print('F9_SYNTHETIC_ENV_BUFFER_LOSS_RESTORE_SMOKE_OK',loss)


class F9A2Probes(unittest.TestCase):
    """A2 teacher task sampler: pure-function and rejection-loop probes (no model)."""

    @staticmethod
    def _task(world, role='anticipatory'):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import EpisodeTaskSpec
        return EpisodeTaskSpec(task_id='t-' + world, world_id='w20-' + world, obstacle_json='/dev/null',
            role=role, seed=1, training_eligible=True,
            identity_hash=hashlib.sha256(world.encode()).hexdigest())

    def test_A2_default_mode_keeps_original_sampler(self):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f9_blocked_sampler import (
            BlockedTeacherTaskSampler)
        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_sampler import TaskSampler
        roles = ('anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery')
        tasks = [self._task('w%d' % i, roles[i % 4]) for i in range(8)]
        first = [TaskSampler(tasks, 7).sample().task_id for _ in range(6)]
        second = [TaskSampler(tasks, 7).sample().task_id for _ in range(6)]
        self.assertEqual(first, second)
        source = Path(__file__).parents[2]/'diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/f9_support.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        text = source.read_text(encoding='utf-8')
        self.assertIn("str(config.f9.get('teacher_task_sampler', 'default'))", text)
        self.assertIn("self.sampler = TaskSampler(factory.tasks, int(config.run.seed) + 91001)", text)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == 'validate_f9'
                            for node in tree.body))

    def test_A2_blocked_rejection_and_library_exclusion(self):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import f9_blocked_sampler as module
        library = (('w0', np.zeros(3), np.array([1., 0., 0.])),)
        self.assertTrue(module.is_library_task('w0', np.zeros(3), np.array([1., 0., 0.]), library))
        self.assertTrue(module.is_library_task('w0', np.full(3, 5e-7), np.array([1., 0., 0.]) + 5e-7, library))
        self.assertFalse(module.is_library_task('w1', np.zeros(3), np.array([1., 0., 0.]), library))
        self.assertFalse(module.is_library_task('w0', np.zeros(3), np.array([1., 0., 1e-5]), library))
        obstacle = None
        from diffusion_policy_3d.env.collision_geom import ObstacleSpec
        obstacle = ObstacleSpec(name='b', shape='sphere', x=0.5, y=0., z=1., radius=0.2)
        start = np.array([0., 0., 1.])
        goal = np.array([1., 0., 1.])
        clearance = module.straight_min_clearance(start, goal, [obstacle])
        # the sampled line passes through the sphere centre: signed clearance is -radius
        self.assertAlmostEqual(clearance, -0.2, places=6)
        self.assertLess(clearance, module.BLOCKED_CLEARANCE_M)
        far = module.straight_min_clearance(start, goal, [ObstacleSpec(
            name='b', shape='sphere', x=0.5, y=5., z=1., radius=0.2)])
        self.assertGreaterEqual(far, module.BLOCKED_CLEARANCE_M)
        self.assertEqual(module.MAX_ATTEMPTS, 500)
        self.assertEqual(module.SAMPLER_SEED, 202611231)
        self.assertEqual(module.STRAIGHT_STEP_M, 0.05)
        sampler = module.BlockedTeacherTaskSampler.__new__(module.BlockedTeacherTaskSampler)
        sampler.accepted, sampler.candidate_attempts = 0, 0
        self.assertIsNone(sampler.acceptance_rate())


if __name__ == '__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--smoke':local_smoke(sys.argv[2])
    else:unittest.main()
