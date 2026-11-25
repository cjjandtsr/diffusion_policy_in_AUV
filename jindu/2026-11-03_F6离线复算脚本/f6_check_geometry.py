"""F6 §5: training-world obstacle files vs dev420 trace geometry.
Usage: python3 -I f6_check_geometry.py <REPO_ROOT> <REPO_ROOT>/jindu/routea_f5_run01/results/eval_dev/full/P6/traces"""
import json, sys, os, hashlib, dataclasses
repo, tracedir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, 'Improved-3D-Diffusion-Policy'))
from diffusion_policy_3d.env.collision_geom import ObstacleSpec
wdir = os.path.join(repo, 'PX4-Autopilot-Tools/sitl_gazebo/worlds/train')
seen = {}
for f in os.listdir(tracedir):
    with open(os.path.join(tracedir, f)) as fh:
        s = json.loads(fh.readline())                      # episode_start record
    seen.setdefault(s['world_id'], (s['obstacle_sha256'], s['geometry']))
norm = lambda L: sorted((dataclasses.astuple(ObstacleSpec.from_dict(o)) for o in L), key=str)
ok_hash = ok_geom = 0
for w, (sha, geom) in sorted(seen.items()):
    raw = open(os.path.join(wdir, w + '.world.obstacles.json'), 'rb').read()
    ok_hash += hashlib.sha256(raw).hexdigest() == sha
    ok_geom += norm(json.loads(raw)['obstacles']) == norm(geom)
print('worlds', len(seen), '| file sha256 == recorded obstacle_sha256:', ok_hash,
      '| ObstacleSpec-parsed geometry identical:', ok_geom)
