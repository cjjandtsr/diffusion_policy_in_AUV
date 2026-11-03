"""Read-only diagnosis of the eval_dev_P8 failure (ValueError: common environment is not v2/H4/8).

Builds the native system exactly as scripts/run_routea_xk_evaluation.py:worker does and prints the
reward-version wiring, without touching the repository or any evaluation artifact.
"""
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--update', type=int, required=True)
parser.add_argument('--source-root', required=True)
args = parser.parse_args()

sys.path.insert(0, str(Path(args.repo).resolve()))
from scripts.run_routea_xk_evaluation import native_system  # noqa: E402

ns = argparse.Namespace(repo=args.repo, source_root=args.source_root,
                        checkpoint=args.checkpoint, expected_update=args.update)
system, config, header = native_system(ns)

print('config.reward_version      =', config.get('reward_version'))
print('system.reward.version      =', getattr(system.reward, 'version', '<no attribute>'))
print('system.reward type         =', type(system.reward).__module__ + '.' + type(system.reward).__name__)
print('header.update_index        =', header['update_index'])
print('header.runtime_identity    =', header['runtime_identity_hash'])
import scripts.evaluate_dppo_uav_formal_endpoint as endpoint  # noqa: E402
print('endpoint reward_version src= getattr(system.reward, "version", "v13")  (line 429)')
print('=> env.reward_version fed by endpoint would be:',
      getattr(system.reward, 'version', 'v13'))
print('gate requires: v2 / n_exec_steps=4 / rl_action_steps=8 / traj_dt=.3 / max_episode_steps=60 / collision_thresh=.1')
