#!/usr/bin/env python3
"""Strategy-only P0/P6 bridge for the unchanged, existing Gazebo planner node."""
import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import xml.etree.ElementTree as ET


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def make_launch(argv):
    p=argparse.ArgumentParser(description='Clone the known DDPM launch; preserve its controller/observation parameters.')
    for name in ('legacy-launch','legacy-node','output','python','repo','source-root','checkpoint','trial-output','task-id'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--policy',choices=['P0','P6'],required=True)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--policy-position-offset',type=float,nargs=3,default=[0.,0.,0.])
    a=p.parse_args(argv)
    assert not Path(a.output).exists()
    tree=ET.parse(a.legacy_launch)
    nodes=[n for n in tree.getroot().iter('node') if n.get('type')=='ml_planner_gazebo.py']
    assert len(nodes)==1
    node=nodes[0]
    node.set('type','routea_gazebo_policy.py')
    node.set('launch-prefix',shlex.quote(a.python))
    values={'repo':a.repo,'source-root':a.source_root,'checkpoint':a.checkpoint,'output':a.trial_output,
            'task-id':a.task_id,'policy':a.policy,'repeat':str(a.repeat),'legacy-node':a.legacy_node}
    node.set('args',' '.join('--'+k+' '+shlex.quote(v) for k,v in values.items())+
             ' --policy-position-offset '+' '.join(map(str,a.policy_position_offset)))
    params={x.get('name'):x for x in node.findall('param')}
    if 'deployment_action_steps' not in params:
        params['deployment_action_steps']=ET.SubElement(node,'param',name='deployment_action_steps')
    params['deployment_action_steps'].set('value',str(a.steps))
    for name in ('waypoint_num','waypoint0_x','waypoint0_y','waypoint0_z'):
        for arg in tree.getroot().findall('arg'):
            if arg.get('name')==name and 'value' in arg.attrib:
                arg.set('default',arg.attrib.pop('value'))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    tree.write(a.output,encoding='utf-8',xml_declaration=True)
    print('GAZEBO_STRATEGY_LAUNCH_READY',a.output)


class PolicyAdapter:
    def __init__(self,args):
        import torch
        self.args=args
        worker=load_file('routea_native_deployment',Path(args.source_root)/'scripts/run_routea_xk_evaluation.py')
        self.system,self.config,self.header=worker.native_system(argparse.Namespace(
            source_root=args.source_root,repo=args.repo,checkpoint=args.checkpoint,expected_update=256))
        assert self.header['runtime_identity_hash']=='d987235a0cbb5ad395bbd6e7979bdb8d7e0cff1303b0611cb462c7d554bbd61f'
        import hashlib
        assert hashlib.sha256((Path(args.checkpoint)/'state.pt').read_bytes()).hexdigest()=='a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43'
        from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import encode_e5_condition
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ObservationSpec
        self.encode,self.spec=encode_e5_condition,ObservationSpec.e5_route_a()
        self.device=torch.device(self.system.bc_runtime.device)
        self.decision=0
        self.output=Path(args.output);self.output.mkdir(parents=True,exist_ok=False)
        self.log=(self.output/'policy.jsonl').open('x',encoding='utf-8')
        self.override=worker.load_override(Path(args.repo))

    def predict_action(self,obs):
        import torch
        import time
        import numpy as np
        from scripts.routea_paired_support.episodes import action_seed
        seed=action_seed(self.args.task_id,self.args.repeat,self.decision)
        values={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in obs.items()}
        values['agent_pos']=values['agent_pos'].clone()
        values['agent_pos'][...,:3]+=torch.tensor(self.args.policy_position_offset,device=self.device)
        assert tuple(values['point_cloud'].shape)==(1,3,4096,3)
        assert tuple(values['agent_pos'].shape)==(1,3,9)
        assert all(torch.isfinite(v).all() for v in values.values())
        # The original numerical encoder is reused directly. Real sensor frames
        # have no synthetic-ray metadata; none is fabricated or marked verified.
        started=time.time()
        with torch.no_grad(),torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
            torch.manual_seed(seed)
            if self.device.type=='cuda':torch.cuda.manual_seed_all(seed)
            condition=self.encode(self.system.bc_runtime.policy,values,self.spec)
            with self.override(self.system.official_runtime,denoised_clip_value=2.0,
                               min_std_vector=[.1]*11,sampling_mode='deterministic'),torch.jit.optimized_execution(False):
                sample=self.system.official_runtime(cond=condition,deterministic=True,return_chain=True,
                    use_base_policy=self.args.policy=='P0')
            full=self.system.bc_runtime.policy.normalizer['action'].unnormalize(sample.trajectories)
            if self.device.type=='cuda':torch.cuda.synchronize()
        assert tuple(full.shape)==(1,16,11) and torch.isfinite(full).all()
        np.savez_compressed(self.output/('observation_%04d.npz'%self.decision),
                           **{k:v.detach().cpu().numpy() for k,v in values.items()})
        self.log.write(json.dumps(dict(decision=self.decision,seed=seed,policy=self.args.policy,
            actual_use_base_policy=self.args.policy=='P0',runtime_identity_hash=self.header['runtime_identity_hash'],
            sensor_source='real ROS observation; synthetic ray metadata not applicable',
            policy_position_offset=self.args.policy_position_offset,
            started_unix=started,finished_unix=time.time(),full_action_16x11=full[0].cpu().tolist(),
            returned_window=[2,10],executed_window=[2,6]))+'\n');self.log.flush()
        self.decision+=1
        return dict(action=full[:,2:10],action_pred=full)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='make-launch':
        make_launch(sys.argv[2:])
        return
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','source-root','checkpoint','output','task-id'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--policy',choices=['P0','P6'],required=True)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--legacy-node')
    p.add_argument('--policy-position-offset',type=float,nargs=3,default=[0.,0.,0.])
    p.add_argument('--probe-npz',help='offline (3-frame or 8-frame unbatched) observation, no ROS startup')
    args,ros_args=p.parse_known_args()
    adapter=PolicyAdapter(args)
    if args.probe_npz:
        import numpy as np
        with np.load(args.probe_npz,allow_pickle=False) as z:
            obs={k:z[k] for k in ('point_cloud','agent_pos')}
        for k,v in obs.items():
            obs[k]=v[-3:][None] if v.ndim==(3 if k=='point_cloud' else 2) else v[:,-3:]
        adapter.predict_action(obs)
        print('GAZEBO_POLICY_PROBE_COMPLETE',args.policy)
        return
    assert args.legacy_node, 'ROS startup requires --legacy-node'
    legacy=load_file('routea_legacy_gazebo',args.legacy_node)
    import rospy
    import torch
    class Planner(legacy.MLPlannerNode):
        def load_diffusion_policy(self):
            self.device=adapter.device
            self.n_obs_steps,self.action_horizon=3,8
            self.obs_history=deque(maxlen=3)
            self.policy=adapter
            self.use_fp16=False
            self.use_synth_pc=False
            self._pc_pinned=self._pos_pinned=None
    sys.argv=[sys.argv[0]]+ros_args
    Planner()
    rospy.spin()


if __name__=='__main__':main()
