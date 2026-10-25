"""Real CPU SynthPointCloud/KinematicUAVEnv fixture, without a neural policy."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class ActualEnvironment(unittest.TestCase):
    def test_fixed_reset_reproduces_arrays_and_h4_stops_at_first_collision(self):
        import numpy as np
        import torch
        from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
        from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter
        from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v2.facade import RewardV2Facade
        from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
        from scripts.routea_paired_support.core import observation_hash
        from scripts.routea_paired_support.runtime import metadata
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d, torch.jit.optimized_execution(False):
            p=Path(d)/"fixture.world.obstacles.json"
            p.write_text(json.dumps({"spawn_offset":[0,0,0],"obstacles":[
                {"name":"wall","shape":"box","x":3.,"y":0.,"z":1.,
                 "size_x":2.,"size_y":2.,"size_z":2.,"yaw":0.}]}),encoding="utf-8")
            reward=RewardV2Facade()
            env=KinematicUAVEnv(world_id=4,world_pool=[("fixture",str(p))],
                synth_pc_factory=lambda path:SynthPointCloud(path,device="cpu"),
                rl_action_steps=8,n_exec_steps=4,max_episode_steps=60,
                disturb_enable=False,exec_noise_scale=0.,mid_state_init_ratio=0.,
                curriculum_stage="C2",episode_role="anticipatory",reward_version="v2")
            adapter=KinematicAdapter(env,reward,reward.initial_state({}),{})
            payload={"episode_id":"fixture-task","world_name":"fixture","role":"anticipatory",
                     "start_pos":[0.,0.,1.],"goal_pos":[12.,0.,1.],"start_yaw":0.,
                     "z_range":[1.,5.],"disturb_enable":False}
            a,info=reset_fixed_episode(env,payload,seed=12345,reset_callable=adapter.reset)
            json.dumps(metadata(info),allow_nan=False)
            b,context=reset_fixed_episode(env,payload,seed=12345,reset_callable=adapter.reset)
            self.assertEqual(observation_hash(a),observation_hash(b))
            self.assertEqual(a["point_cloud"].shape,(8,4096,3))
            self.assertEqual(a["agent_pos"].shape,(8,9))
            full=torch.zeros((1,16,11),dtype=torch.float32)
            full[:,2:10,0]=1.
            from scripts.routea_paired_support.recovery import continue_flight, clone_snapshot
            from types import SimpleNamespace
            snapshot=(adapter,b,context)
            from scripts.routea_paired_support.mechanism import snapshot_digest, fork_after_first
            digest=snapshot_digest(snapshot)
            self.assertEqual(snapshot_digest(clone_snapshot(snapshot)),digest)
            stationary=dict(full_action_16x11=np.zeros((16,11),np.float32))
            fork,reason=fork_after_first(snapshot,stationary)
            self.assertIsNone(reason)
            self.assertEqual(fork[0].environment.episode_step,1)
            self.assertNotEqual(snapshot_digest(fork),digest)
            self.assertEqual(snapshot_digest(snapshot),digest)
            # First draw must be reused, with the original clock and exact H4 collision.
            no_resample=SimpleNamespace(predict=lambda *a,**k: self.fail('resampled first action'))
            first=dict(full_action_16x11=full[0].numpy(),initial_noise_hash='fixture',observation_provenance={})
            result=continue_flight(no_resample,snapshot,'fixture',0,first)
            self.assertEqual(result['outcome'],'collision')
            self.assertEqual(len(result['rows']),1)
            self.assertEqual(env.episode_step,0)
            late=clone_snapshot(snapshot)
            late[0].environment.episode_step=59
            first['full_action_16x11']=np.zeros((16,11),np.float32)
            result=continue_flight(no_resample,late,'fixture',0,first)
            self.assertEqual(result['outcome'],'time_limit')
            self.assertEqual(result['remaining_decisions'],0)
            self.assertEqual(len(result['rows']),1)
            _,_,terminated,truncated,info=adapter.step(ActionBundle.from_full(full))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["done_reason"],"collision")
            self.assertLess(info["executed_time_s"],1.2)
            self.assertLess(info["executed_path"][-1,0],2.)
            json.dumps(metadata(info),allow_nan=False)


if __name__=="__main__":
    unittest.main()
