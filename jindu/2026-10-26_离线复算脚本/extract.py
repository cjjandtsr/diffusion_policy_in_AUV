import json, glob, sys, os, ast
B = '/projects/sandbox/diffusion_policy_in_AUV/jindu/routea_common_a1_v1_run01/evaluation/full/'
out = {}
for pol in ['bc', 'rl_391a6656a0f7_u32', 'rl_391a6656a0f7_u128', 'rl_391a6656a0f7_u256']:
    for f in glob.glob(B + pol + '/traces/*.jsonl'):
        ep = dict(path=[], yaw=[], dec_pos=[])
        for line in open(f):
            x = json.loads(line)
            t = x.get('type')
            if t == 'episode_start':
                ep.update(task=x['task_id'], rep=x['repeat'], role=x['role'], geom=x['geometry'],
                          start=x['task']['start_pos'] if 'start_pos' in x['task'] else None,
                          goal=x['task'].get('goal_pos'), taskd={k: v for k, v in x['task'].items() if k in ('start_pos', 'goal_pos', 'start_yaw')})
            elif t == 'transition':
                ep['dec_pos'].append(x['pre_state']['position'])
                ep['yaw'].append(x['pre_state']['yaw'])
                ep['path'].append(x['executed_path'])
                ep['last_reason'] = x['done_reason']
            elif t == 'episode_end':
                ep['done'] = x['done_reason']; ep['tm'] = x.get('trajectory_metrics')
        key = (pol.split('_')[-1], ep['task'], str(ep['rep']))
        out['|'.join(key)] = ep
    print(pol, len(out), flush=True)
json.dump(out, open('/projects/sandbox/scratch/eps.json', 'w'))
