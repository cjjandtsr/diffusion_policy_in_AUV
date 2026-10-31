"""Read-only audit of returned first-H4 crossover and target-response records."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def aggregate(rows):
    if not rows:
        return {'n': 0}
    result = dict(n=len(rows), states=len({r['state'] for r in rows}),
                  tasks=len({r['task'] for r in rows}))
    for k in ('projection','progress','minball','cosine','yaw_error_reduction','length','forward'):
        values=[r[k] for r in rows if r[k] is not None]
        result[k+'_mean']=float(np.mean(values)) if values else None
    for k in ('positive','near_zero','h4_reached','h8_reached','h4_collision','h8_collision'):
        result[k]=sum(r[k] for r in rows)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); root=args.snapshot; p=root/'diagnostic/mechanism'
    hashes=[]
    for line in (root/'return_sha256.txt').read_text().splitlines():
        h,name=line.split('  ',1); file=root/name
        assert hashlib.sha256(file.read_bytes()).hexdigest()==h,name
        hashes.append(name)
    policies=sorted(x.name for x in (p/'diagonal').iterdir())
    diagonal=0
    for policy in policies:
        for file in (p/'diagonal'/policy).glob('*.json'):
            sid,draw=file.stem.rsplit('_',1); draw=int(draw)
            actual=read(file); old=read(args.baseline/policy/(sid+'.json'))
            assert actual['sample']==old['samples'][draw],file
            assert actual['continuation']==next(x for x in old['continuations'] if x['draw']==draw),file
            diagonal+=1
    cross=read(p/'crossed_cells.json'); rows=[d for s in cross['states'] for d in s['draws']]
    groups={}
    for group in sorted({r['group'] for r in rows}):
        rr=[r for r in rows if r['group']==group]
        groups[group]=dict(n=len(rr), outcomes={k:dict(Counter(r['outcomes'][k] for r in rr)) for k in 'ABCD'},
            patterns=dict(Counter(''.join('1' if r['outcomes'][k]=='reached' else '0' for k in 'ABCD') for r in rr)))
    crossfiles=list((p/'crossed').glob('*/*.json'))
    for file in crossfiles:
        r=read(file); sid=r['state_id']; draw=r['draw']; source=r['source_policy']
        original=read(p/'diagonal'/source/(sid+'_%d.json'%draw))['continuation']
        if r['absorbed']:
            assert r['continuation'] is None and original['rows'][0]['done_reason']==r['outcome']
        else:
            first=r['continuation']['rows'][0]; old=original['rows'][1]
            for key in ('seed','initial_noise_hash','observation_provenance','pre_state','decision','pointcloud_metadata_before'):
                assert first[key]==old[key],(file,key)
            assert first['cycle']==1 and len(r['continuation']['rows'])<=11
    goalrows=[]; pairs={}; max_error=0.; candidates=read(p/'goal_candidates.json')
    invalid=[dict(state=s,**c) for s,cs in candidates.items() for c in cs if not c['valid']]
    for file in (p/'goals').glob('*/*.json'):
        obj=read(file); policy='bc' if obj['policy']=='bc' else obj['policy'].rsplit('_',1)[1]
        for item in obj['results']:
            c=item['candidate']
            if not c['valid']:
                assert not item['samples']; continue
            hist=np.array(item['agent_pos_history']); goal=np.array(c['goal'])
            # Reconstruct one common frame translation from world path and final odom position.
            for sample in item['samples']:
                path=np.array(sample['planned_future_path']); offset=path[0]-hist[-1,:3]
                yaw=np.arctan2(hist[:,7],hist[:,8]); delta=goal-(hist[:,:3]+offset)
                expected=np.stack([np.cos(yaw)*delta[:,0]+np.sin(yaw)*delta[:,1],
                    -np.sin(yaw)*delta[:,0]+np.cos(yaw)*delta[:,1],delta[:,2]],axis=1)
                max_error=max(max_error,float(np.abs(expected-hist[:,3:6]).max()),
                    float(np.abs(np.linalg.norm(delta,axis=1)-hist[:,6]).max()))
                direction=goal-path[0]; disp=path[4]-path[0]
                norm=np.linalg.norm(disp); projection=float(disp@direction/np.linalg.norm(direction))
                assert abs(projection-sample['h4_target_projection_m'])<1e-10
                for h in (4,8):
                    a=path[:h]; d=np.diff(path[:h+1],axis=0)
                    t=np.clip(((goal-a)*d).sum(1)/np.maximum((d*d).sum(1),1e-30),0,1)
                    minimum=float(np.linalg.norm(a+t[:,None]*d-goal,axis=1).min())
                    assert abs(minimum-sample['h%d_min_goal_center_distance_m'%h])<1e-10
                wrap=lambda x: (x+np.pi)%(2*np.pi)-np.pi
                bearing=float(np.arctan2(direction[1],direction[0])-yaw[-1]); bearing=wrap(bearing)
                yawchange=float(np.asarray(sample['full_action_16x11'])[2:6,9].sum())
                reduction=np.rad2deg(abs(bearing)-abs(wrap(bearing-yawchange)))
                # Initial bearing held fixed: response to target, not final tracking error.
                r=dict(policy=policy,state=obj['state_id'],task=obj['task_id'],group=obj['group'],
                    label=c['label'],angle=c['bearing_deg'],clear=c['direct_clear'],draw=sample['draw'],
                    projection=projection,progress=float(np.linalg.norm(direction)-np.linalg.norm(goal-path[4])),
                    minball=sample['h4_min_goal_ball_distance_m'],cosine=float(projection/norm) if norm>1e-6 else None,
                    positive=projection>0,near_zero=bool(norm<=1e-6),yaw_error_reduction=float(reduction),
                    length=float(norm),forward=float(disp[:2]@np.array([np.cos(yaw[-1]),np.sin(yaw[-1])])),
                    bearing_actual_deg=float(np.rad2deg(bearing)))
                for h in (4,8):
                    event=sample['metrics']['h%d_first_event'%h]
                    r['h%d_reached'%h]=bool(event and event['kind']=='reached')
                    r['h%d_collision'%h]=bool(event and event['kind']!='reached')
                key=(obj['state_id'],c['label'],sample['draw']); pairs.setdefault(key,{})[policy]=(r,sample)
                if c['label']=='original':
                    original=read(p/'diagonal'/obj['policy']/(obj['state_id']+'_%d.json'%sample['draw']))['sample']
                    assert sample['full_action_16x11']==original['full_action_16x11']
                    assert sample['observation_provenance']==original['observation_provenance']
                goalrows.append(r)
    assert max_error<1e-4,max_error
    for key,values in pairs.items():
        assert set(values)=={'bc','u128','u256'}
        for policy in ('u128','u256'):
            assert values[policy][1]['initial_noise_hash']==values['bc'][1]['initial_noise_hash']
            assert values[policy][1]['observation_provenance']==values['bc'][1]['observation_provenance']
    stats={}
    for subset in ('all','original','changed_clear','changed_blocked'):
        rs=[r for r in goalrows if subset=='all' or (subset=='original' and r['label']=='original') or
            (subset.startswith('changed') and r['label']!='original' and r['clear']==(subset=='changed_clear'))]
        stats[subset]={policy:aggregate([r for r in rs if r['policy']==policy]) for policy in ('bc','u128','u256')}
    angles={str(a):{policy:aggregate([r for r in goalrows if r['angle']==a and r['clear'] and r['policy']==policy])
                   for policy in ('bc','u128','u256')} for a in (-135,-90,-45,0,45,90,135,180)}
    paired=[]
    for (sid,label,draw),v in pairs.items():
        b,x,y=[v[n][0] for n in ('bc','u128','u256')]
        paired.append(dict(state=sid,task=x['task'],group=x['group'],label=label,draw=draw,clear=x['clear'],
            u256_minus_u128_projection=y['projection']-x['projection'],
            u128_minus_bc_projection=x['projection']-b['projection'],
            u256_minus_bc_projection=y['projection']-b['projection']))
    buckets={label:{policy:aggregate([r for r in goalrows if r['angle'] in angles and r['policy']==policy])
             for policy in ('bc','u128','u256')} for label,angles in
             [('front',[0]),('front_side',[-45,45]),('side',[-90,90]),('rear',[-135,135,180])]}
    state_rear=[]
    for sid in sorted({r['state'] for r in goalrows}):
        state_rear.append(dict(state=sid,values={policy:aggregate([r for r in goalrows if r['state']==sid and
            r['angle'] in [-135,135,180] and r['policy']==policy]) for policy in ('bc','u128','u256')}))
    original_cycles=[]
    geometry={s['id']:[ObstacleSpec.from_dict(o) for o in s['header']['geometry']]
              for s in read(p/'frozen_bank.json')['states']}
    for policy in policies:
        for file in (p/'diagonal'/policy).glob('*.json'):
            sid=file.stem.rsplit('_',1)[0]; goal=np.array(candidates[sid][0]['goal'])
            flight=read(file)['continuation']
            for row in flight['rows']:
                pos=np.array(row['pre_state']['position']); yaw=row['pre_state']['yaw']; delta=goal-pos
                angle=(np.arctan2(delta[1],delta[0])-yaw+np.pi)%(2*np.pi)-np.pi
                path=np.array(row['planned_future_path']); disp=path[4]-path[0]
                original_cycles.append(dict(policy=policy,state=sid,draw=flight['draw'],cycle=row['cycle'],
                    outcome=flight['outcome'],angle_deg=float(np.rad2deg(angle)),
                    direct_clear=segment_obstacle_collision(pos,goal,geometry[sid],.1) is None,
                    projection=float(disp@delta/np.linalg.norm(delta)),
                    progress=float(np.linalg.norm(delta)-np.linalg.norm(goal-path[4]))))
    bc_candidates=[]
    for file in (args.baseline/'bc').glob('*.json'):
        data=read(file); samples=data['samples']
        bc_candidates.append(dict(state=file.stem,n=len(samples),
            h4_positive=sum(s['metrics']['h4_progress']>0 for s in samples),
            h8_reached=sum(bool(s['metrics']['h8_first_event'] and s['metrics']['h8_first_event']['kind']=='reached') for s in samples)))
    output=dict(audit=dict(manifest_files=len(hashes),diagonal_exact=diagonal,crossed=len(crossfiles),
        paired_goal_inputs=len(pairs),history_target_max_error=max_error,invalid_candidates=invalid),
        groups=groups,task_equal=cross['task_equal_arrival_fraction'],goal_stats=stats,clear_angles=angles,
        buckets=buckets,state_rear=state_rear,original_cycles=original_cycles,bc_candidates=bc_candidates,
        goal_rows=goalrows,paired_projection=paired)
    args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k in ('audit','groups','task_equal','buckets')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
