#!/usr/bin/env python3
"""
Generate fixed episode_configs for deterministic paired evaluation.
One JSON per split: rl_val, rl_test, layerA.
Each world gets 4 configs with fixed seed → pre/post eval reuse same start/goal.

Design ref: program/review/GPT_Review_StartGoal_Sampling_Design_20260709.md §2.3
"""
import json, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
    load_routea_rl_admission_manifest, build_role_buckets)

MASTER_SEED = 42
N_CONFIGS_PER_WORLD = 4
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'eval_configs'
WORLDS_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds')
# Split → subdirectory mapping (post-2026-07-25 reorganization)
SPLIT_SUBDIR = {'rl_val': 'val', 'rl_test': 'test', 'rl_train': 'train'}
MANIFEST_PATH = os.path.expanduser(
    '~/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest_kinematic_v1.json')
LAYER_A_DIR = os.path.join(WORLDS_DIR, 'ood_experiment_worlds', 'layerA')
CURRICULUM = 'C2'


def generate_for_role_split(split_name, role_buckets):
    """Generate configs for worlds in admission manifest split (rl_val / rl_test)."""
    configs = {}
    rng = np.random.default_rng(MASTER_SEED)

    for role, entries in role_buckets.items():
        for entry in entries:
            uid = entry.world_uid
            subdir = SPLIT_SUBDIR.get(split_name, 'train')
            json_path = os.path.join(WORLDS_DIR, subdir, f'{uid}.world.obstacles.json')

            if not os.path.exists(json_path):
                print(f'  [SKIP] missing: {json_path}')
                continue

            synth_pc = SynthPointCloud(obstacles_json=json_path)
            world_configs = []

            for i in range(N_CONFIGS_PER_WORLD):
                # Per-config seed = master_seed + hash of world + config index
                cfg_seed = int(MASTER_SEED * 10000 + abs(hash(uid + str(i))) % (2**20))
                env = KinematicUAVEnv(
                    world_pool=[(uid, json_path)],
                    synth_pc_pool={uid: synth_pc},
                    curriculum_stage=CURRICULUM,
                    max_episode_steps=200,
                    episode_role=role,          # ★ use world's actual role
                    disturb_enable=False,       # ★ eval: no disturbance
                    mid_state_init_ratio=0.0,   # ★ eval: no mid-state init
                    seed=cfg_seed)

                obs_dict, _ = env.reset(seed=cfg_seed)
                cfg = env.export_episode_config(f'{uid}__cfg{i:02d}')
                cfg['role'] = role
                world_configs.append(cfg)
                env.close()

            configs[uid] = world_configs
            print(f'  [{role}] {uid}: {N_CONFIGS_PER_WORLD} configs')

    return configs


def generate_for_layera():
    """Generate configs for layerA worlds (no role labels, default anticipatory)."""
    configs = {}
    rng = np.random.default_rng(MASTER_SEED)

    layer_a_worlds = sorted([
        f.replace('.obstacles.json', '')
        for f in os.listdir(LAYER_A_DIR) if f.endswith('.obstacles.json')
    ])

    for uid in layer_a_worlds:
        json_path = os.path.join(LAYER_A_DIR, f'{uid}.obstacles.json')
        synth_pc = SynthPointCloud(obstacles_json=json_path)
        world_configs = []

        for i in range(N_CONFIGS_PER_WORLD):
            cfg_seed = int(MASTER_SEED * 10000 + abs(hash(uid + str(i))) % (2**20))
            env = KinematicUAVEnv(
                world_pool=[(uid, json_path)],
                synth_pc_pool={uid: synth_pc},
                curriculum_stage=CURRICULUM,
                max_episode_steps=200,
                episode_role='anticipatory',   # layerA: default role
                disturb_enable=False,
                mid_state_init_ratio=0.0,
                seed=cfg_seed)

            obs_dict, _ = env.reset(seed=cfg_seed)
            cfg = env.export_episode_config(f'{uid}__cfg{i:02d}')
            cfg['role'] = 'anticipatory'
            world_configs.append(cfg)
            env.close()

        configs[uid] = world_configs
        print(f'  [layerA] {uid}: {N_CONFIGS_PER_WORLD} configs')

    return configs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=== Generating fixed episode_configs for paired eval ===')
    print(f'Master seed: {MASTER_SEED}, configs/world: {N_CONFIGS_PER_WORLD}\n')

    # --- rl_val ---
    print('--- rl_val ---')
    manifest = load_routea_rl_admission_manifest(MANIFEST_PATH)
    rl_val_buckets = build_role_buckets(manifest, 'rl_val')
    rl_val_configs = generate_for_role_split('rl_val', rl_val_buckets)
    out_path = OUTPUT_DIR / 'episode_configs_rl_val.json'
    with open(out_path, 'w') as f:
        json.dump(rl_val_configs, f, indent=2)
    n_worlds = len(rl_val_configs)
    print(f'  Saved {n_worlds} worlds → {out_path}\n')

    # --- rl_test ---
    print('--- rl_test ---')
    rl_test_buckets = build_role_buckets(manifest, 'rl_test')
    rl_test_configs = generate_for_role_split('rl_test', rl_test_buckets)
    out_path = OUTPUT_DIR / 'episode_configs_rl_test.json'
    with open(out_path, 'w') as f:
        json.dump(rl_test_configs, f, indent=2)
    n_worlds = len(rl_test_configs)
    print(f'  Saved {n_worlds} worlds → {out_path}\n')

    # --- rl_train ---
    print('--- rl_train ---')
    rl_train_buckets = build_role_buckets(manifest, 'rl_train')
    rl_train_configs = generate_for_role_split('rl_train', rl_train_buckets)
    out_path = OUTPUT_DIR / 'episode_configs_rl_train.json'
    with open(out_path, 'w') as f:
        json.dump(rl_train_configs, f, indent=2)
    n_worlds = len(rl_train_configs)
    print(f'  Saved {n_worlds} worlds → {out_path}\n')

    # --- layerA ---
    print('--- layerA ---')
    layera_configs = generate_for_layera()
    out_path = OUTPUT_DIR / 'episode_configs_layerA.json'
    with open(out_path, 'w') as f:
        json.dump(layera_configs, f, indent=2)
    n_worlds = len(layera_configs)
    print(f'  Saved {n_worlds} worlds → {out_path}\n')

    print('=== Done ===')


if __name__ == '__main__':
    main()
