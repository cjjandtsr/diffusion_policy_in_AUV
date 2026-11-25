"""A-level arithmetic/behavior probes; native P6 replay is a separate WSL check."""
import importlib.util
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.routea_denoising_guidance import action_positions, body_to_world, costs, Guidance, install
from scripts.run_routea_e3g_probe import e3_verdict, guided_verdict
from scripts.routea_paired_support.core import action_views
from diffusion_policy_3d.env import kinematic_uav_env as kenv
from diffusion_policy_3d.env.traj_utils import action_to_world
from diffusion_policy_3d.env.collision_geom import ObstacleSpec


EVIDENCE = Path(os.environ.get('E3G_EVIDENCE_ROOT', str(Path(__file__).resolve().parents[4])))


def local_smoke(output):
    """One real environment/record/load transaction with a labelled synthetic actor."""
    from dataclasses import dataclass
    from scripts.routea_paired_support.episodes import record_episode, load_commit
    from scripts.routea_paired_support.core import observation_hash, read_json, write_json
    from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v2.facade import RewardV2Facade
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter
    env, _ = environment('E3')
    env.goal_w = np.array([1.5, 0., 1.], np.float32)
    env._reward_history = dict(initial_distance=1.5, best_distance=1.5)
    facade = RewardV2Facade()
    adapter = KinematicAdapter(env, facade, facade.initial_state())
    full = np.zeros((1, 16, 11), np.float32)
    full[:, :, 0] = .5
    @dataclass
    class Provenance:
        purpose: str = 'synthetic actor; no native P6 model loaded'
    policy = SimpleNamespace(use_base_policy=False, identity_hash='SYNTHETIC_NO_MODEL',
        last_provenance=Provenance(), predict=lambda *a, **k: SimpleNamespace(
            full=full, policy=full[:, 2:10], rollout_executed=full[:, 2:6]))
    obs = dict(point_cloud=np.ones((3, 4096, 3), np.float32), agent_pos=np.zeros((3, 9), np.float32))
    context = dict(pointcloud_meta_history=[])
    payload = dict(start_pos=[0., 0., 1.], goal_pos=[1.5, 0., 1.], z_range=[0., 10.])
    task = dict(payload=payload, role='anticipatory')
    entry = dict(reset_seed=1, observation_hash=observation_hash(obs), key='synthetic_h2',
                 task_id='synthetic', repeat=1, initial_hash='synthetic')
    world = dict(world_id='synthetic', geometry=[], obstacle_sha256='synthetic', spawn_offset=[0., 0., 0.])
    with patch('scripts.evaluate_dppo_uav_formal_endpoint.reset_fixed_episode', return_value=(obs, context)):
        record_episode(output, 'SYNTHETIC_NO_MODEL', policy, adapter, task, entry, world, 'synthetic')
    h, rows, summary = load_commit(output, Path(output) / 'commits/synthetic_h2.json')
    assert summary['safe_arrival'] and h['evaluation_protocol'] == 'E3'
    write_json(Path(output) / 'local_smoke.json', dict(passed=True, actual_environment=True,
        native_model_loaded=False, synthetic_actor=True, decisions=len(rows), summary=summary))
    print('LOCAL_E3_RECORD_LOAD_SMOKE_OK', len(rows), 'SYNTHETIC_NO_MODEL')


def environment(protocol):
    path = Path(__file__).parents[1] / 'dppo_uav_rebuild/rewards/test_reward_v2_integration.py'
    spec = importlib.util.spec_from_file_location('e3g_existing_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pc = module.PointCloud()
    with patch.object(kenv, 'load_obstacles', return_value=[]), patch.object(
            kenv.KinematicUAVEnv, '_load_spawn_offset', staticmethod(lambda _: np.zeros(3))):
        env = kenv.KinematicUAVEnv(synth_pc=pc, reward_version='v2',
            goal_world=np.array([20., 0., 1.], np.float32), disturb_enable=False,
            n_exec_steps=2 if protocol == 'E3' else 4,
            max_episode_steps=120 if protocol == 'E3' else 60, evaluation_protocol=protocol)
    env._drone_pos = np.array([0., 0., 1.], np.float32)
    env._drone_vel = np.zeros(3, np.float32)
    env._drone_yaw = 0.
    env._episode_start_world = env._drone_pos.copy()
    env._episode_identity_hash = 'e' * 64
    env._observation_frame_index = 7
    env._episode_rng_seed = 1
    env._reward_history = dict(initial_distance=20., best_distance=20.)
    env.exec_noise_scale = 0.
    return env, pc


class E3GProbes(unittest.TestCase):
    def test_A1_A2_h2_free_and_slice(self):
        env, pc = environment('E3')
        full = np.zeros((16, 11), np.float32)
        full[2:, 0] = 1.
        full[2:, 3] = np.arange(14) + 1
        full[2:, 9] = .2
        action_views(full, full[2:10], full[2:4], 2)
        _, _, terminal, truncated, info = env.step(full[2:10])
        np.testing.assert_array_equal(env._drone_pos, [2, 0, 1])
        np.testing.assert_array_equal(env._drone_vel, [2, 0, 0])
        self.assertAlmostEqual(env._drone_yaw, .4, places=6)
        self.assertEqual(pc.calls, 2)
        self.assertAlmostEqual(info['executed_time_s'], .6)
        self.assertFalse(terminal or truncated)

    def test_A1_first_second_segment_collision(self):
        from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
        # Radius .1, contact .1 => front surface at x-.2.
        for center, segment, end, yaw, velocity in [( .7, 0, .5, .1, .5), (1.7, 1, 1.5, .3, 1.5)]:
            env, pc = environment('E3')
            env.obstacles = [ObstacleSpec(name='probe', shape='sphere', x=center, y=0., z=1., radius=.1)]
            action = np.zeros((8, 11), np.float32)
            action[:, 0] = 1.
            action[:, 3] = np.arange(8) + 1
            action[:, 9] = .2
            _, _, terminal, _, info = env.step(action)
            self.assertTrue(terminal)
            self.assertEqual(info['event_segment'], segment)
            self.assertAlmostEqual(env._drone_pos[0], end, places=5)
            self.assertAlmostEqual(env._drone_yaw, yaw, places=5)
            self.assertAlmostEqual(env._drone_vel[0], velocity, places=5)
            self.assertEqual(pc.calls, segment + 1)

    def test_A1_e2_original_transaction(self):
        # Execute the preserved BEFORE function on an equivalent real environment.
        path = EVIDENCE / 'program/code_backups/20261120_e3g_implementation_before/diffusion_policy_3d/env/reward_v2_support/environment.py'
        spec = importlib.util.spec_from_file_location('diffusion_policy_3d.env.reward_v2_support.e3g_before', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a, _ = environment('E2')
        b, _ = environment('E2')
        action = np.zeros((8, 11), np.float32)
        action[:, 0] = .25
        old, new = module.step_v2(a, action), b.step(action)
        np.testing.assert_array_equal(a._drone_pos, b._drone_pos)
        self.assertEqual(old[1:4], new[1:4])
        for key in ('executed_path', 'executed_time_s', 'reward_terms', 'done_reason'):
            np.testing.assert_equal(old[4][key], new[4][key])
        for key in old[0]:
            np.testing.assert_array_equal(old[0][key], new[0][key])

    def test_A3_world_position_equivalence(self):
        x = torch.arange(16 * 11, dtype=torch.float32).reshape(1, 16, 11) / 1000
        pos = np.array([1., 2., 3.], np.float32)
        value = action_positions(x, lambda a: a, pos)[0].numpy()
        expected = action_to_world(x[0, 2:].numpy(), pos, np.zeros(3, np.float32), .4)
        np.testing.assert_array_equal(value, expected[:, :3])
        np.testing.assert_allclose(value[-1], pos + x[0, 2:, :3].sum(0).numpy(), atol=1e-5)

    def test_A4_collision_ground_and_world(self):
        p = torch.tensor([[.2, 0., 1.], [2., 0., 1.]], requires_grad=True)
        cloud = torch.tensor([[0., 0., 1.], [4., 0., 1.], [6., 0., 1.]])
        col, _ = costs(p, cloud)
        self.assertAlmostEqual(float(col), .16, places=6)
        g, = torch.autograd.grad(col, p)
        self.assertAlmostEqual(float(g[0, 0]), -.8, places=6)
        ground, _ = costs(torch.tensor([[0., 0., .2]]), torch.empty((0, 3)))
        self.assertAlmostEqual(float(ground), .09, places=6)
        w = body_to_world(torch.tensor([[1., 0., 0.]]), [2, 3, 1], np.pi / 2)
        np.testing.assert_allclose(w.numpy(), [[2, 4, 1]], atol=1e-6)

    def test_A5_overlap_and_reset(self):
        previous = torch.arange(14.).unsqueeze(1).repeat(1, 3)
        current = torch.zeros(14, 3)
        current[:10] = previous[4:14] + 1.
        _, con = costs(current, torch.empty(0, 3), previous)
        self.assertEqual(float(con), 3.)
        self.assertEqual(float(costs(current, torch.empty(0, 3))[1]), 0.)

    def test_A6_step_gate_updates_clip_zero(self):
        guide = Guidance(lambda x: x, .2, 0)
        guide.begin(torch.tensor([[.2, 0., 0.]]), [0., 0., 1.], 0.)
        x = torch.zeros(1, 16, 11)
        self.assertIs(guide.update(x, torch.tensor([6])), x)
        value = guide.update(x, torch.tensor([7]))
        self.assertEqual(guide.update_count, 2)
        self.assertLessEqual(float(value.abs().max()), 2.)
        self.assertFalse(torch.equal(x, value))
        guide.begin(torch.tensor([[10., 10., 0.]]), [0., 0., 1.], 0.)
        self.assertIs(guide.update(x, torch.tensor([9])), x)

    def test_A8_verdict_boundaries(self):
        self.assertEqual(e3_verdict(-15, 0, -4), '执行周期是有效杠杆')
        self.assertEqual(e3_verdict(-15, 0, -5), '不确定')
        self.assertEqual(e3_verdict(-15, -1, 0), '不确定')
        self.assertEqual(e3_verdict(-14, 20, 20), '执行周期在本探针下不是主要杠杆')
        self.assertEqual(guided_verdict(49, 10, 237, 123), '引导有效')
        self.assertEqual(guided_verdict(49, 9, 237, 123), '不确定')
        self.assertEqual(guided_verdict(48, 20, 255, 0), '本探针下无效')

    def test_A6_native_epsilon_mean_and_zero_cost(self):
        import types
        path = Path(__file__).parents[2] / 'third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py'
        node = next(n for n in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
                    if isinstance(n, ast.FunctionDef) and n.name == 'p_mean_var')
        def extract(a, index, shape):
            return a[index].reshape(-1, 1, 1)
        namespace = dict(torch=torch, extract=extract)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        runtime = SimpleNamespace(use_ddim=True, predict_epsilon=False, ddim_steps=10,
            ft_denoising_steps=10, recalculate_epsilon_after_clip=False, denoised_clip_value=2.,
            eps_clip_value=None, native_ddim_cpu_scalar_compat=False, native_ddim_pow_compat=False,
            ddim_alphas=torch.full((10,), .5), ddim_alphas_prev=torch.full((10,), .8),
            ddim_sqrt_one_minus_alphas=torch.full((10,), .5 ** .5),
            actor=lambda x, t, cond: torch.full_like(x, 3.),
            actor_ft=lambda x, t, cond: torch.full_like(x, 3.))
        runtime.p_mean_var = types.MethodType(namespace['p_mean_var'], runtime)
        x = torch.ones(1, 16, 11)
        kwargs = dict(x=x, t=torch.tensor([0]), cond={'state': torch.ones(1, 1)},
                      index=torch.tensor([9]), deterministic=True)
        original = runtime.p_mean_var(**kwargs)
        epsilon = (1. - (.5 ** .5) * 3.) / (.5 ** .5)
        expected = (.8 ** .5) * 2. + (.2 ** .5) * epsilon
        self.assertAlmostEqual(float(original[0][0, 0, 0]), expected, places=6)
        guide = Guidance(lambda a: a, .1, 0)
        guide.begin(torch.empty(0, 3), [0., 0., 1.], 0.)
        identity = install(runtime, guide)
        self.assertIn('injected_function_sha256', identity)
        unchanged = runtime.p_mean_var(**kwargs)
        for a, b in zip(original, unchanged):
            self.assertTrue(torch.equal(a, b))
        guide.update = lambda x0, index: x0 - .1
        guided = runtime.p_mean_var(**kwargs)
        self.assertAlmostEqual(float(guided[0][0, 0, 0]), expected - (.8 ** .5) * .1, places=6)

    def test_A10_h2_timeout_fields_and_validator(self):
        from scripts.routea_paired_support.episodes import validate_rows
        from scripts.routea_paired_support.core import trace_metrics
        env, _ = environment('E3')
        full = np.zeros((16, 11), np.float32)
        header = dict(evaluation_protocol='E3', n_exec_steps=2, decision_period_s=.6,
                      timeout_decision_index=119, geometry=[], terminate_at_space_limits=False,
                      task=dict(start_pos=[0., 0., 1.], goal_pos=[20., 0., 1.], z_range=[0., 10.]))
        rows = []
        for i in range(120):
            before = dict(position=env._drone_pos.tolist(), velocity=env._drone_vel.tolist(), yaw=float(env._drone_yaw))
            _, _, terminal, truncated, info = env.step(full[2:10])
            rows.append(dict(evaluation_protocol='E3', n_exec_steps=2, decision_period_s=.6,
                timeout_decision_index=119, decision=i, pre_state=before,
                post_state=dict(position=env._drone_pos.tolist()), full_action_16x11=full.tolist(),
                policy_action_8x11=full[2:10].tolist(), executed_action_2x11=full[2:4].tolist(),
                candidate_h2_path=[[0., 0., 1.]] * 3, executed_path=info['executed_path'].tolist(),
                executed_time_s=info['executed_time_s'], done_reason=info['done_reason'],
                event_segment=info['event_segment'], event_fraction=info['event_fraction']))
        self.assertFalse(terminal)
        self.assertTrue(truncated)
        self.assertEqual(rows[-1]['done_reason'], 'time_limit')
        summary = dict(status='valid', done_reason='time_limit', safe_arrival=False, decisions=120,
                       physical_time_s=72., trajectory_metrics=trace_metrics([[0., 0., 1.]], [20., 0., 1.]))
        validate_rows(header, rows, summary)
        rows[-1]['timeout_decision_index'] = 59
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_rows(header, rows, summary)

    def test_A7_no_privileged_map_access(self):
        import inspect
        import scripts.routea_denoising_guidance as guidance
        tree = ast.parse(inspect.getsource(guidance))
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse({'obstacles', 'obstacle_path', 'world_pool'} & attributes)
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertFalse(any('.obstacles.json' in value for value in strings))

    def test_A8_first_entry_and_blocked_identity(self):
        from scripts.f7_analysis import outcome_1m
        from scripts.routea_paired_support.core import read_json, file_hash
        root = EVIDENCE
        path = root / 'sheji/2026-11-17_F8中间点规则/blocked_tasks_dev420.json'
        self.assertEqual(file_hash(path), '32cbc908b61c899a9ad2d55978857de3eb6dfa752182411be7dcf7b12bd2ab3d')
        self.assertEqual(len(read_json(path)), 33)
        self.assertEqual(outcome_1m([dict(executed_path=[[2., 0., 0.], [.8, 0., 0.]])], np.zeros(3), 'collision'), 'reached')
        self.assertEqual(outcome_1m([dict(executed_path=[[2., 0., 0.], [1.2, 0., 0.]])], np.zeros(3), 'collision'), 'collision')

    def test_A10_existing_e2_raw_validation_identical(self):
        from scripts.routea_paired_support.episodes import load_commit
        root = EVIDENCE
        before = root / 'program/code_backups/20261120_e3g_implementation_before/scripts/routea_paired_support/episodes.py'
        spec = importlib.util.spec_from_file_location('scripts.routea_paired_support.e3g_before_episodes', before)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ('dev420', 'val20'):
            job = root / ('wsl_snapshot/routea_p6_u256_e2_' + name + '_run01/full/P6')
            commit = next(iter(sorted((job / 'commits').glob('*.json'))))
            self.assertEqual(module.load_commit(job, commit), load_commit(job, commit))


if __name__ == '__main__':
    unittest.main()
