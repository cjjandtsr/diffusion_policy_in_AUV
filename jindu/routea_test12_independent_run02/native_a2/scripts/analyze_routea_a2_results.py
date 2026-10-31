"""Read-only A2 return audit and paired endpoint recomputation."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision

PROJECT=ROOT.parent.parent
RETURN=PROJECT/'wsl_snapshot/routea_a2_selective_keep_run01'
OUT=PROJECT/'jindu/2026-10-26_A2独立复算_v1_0.json'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def pair(a,b):
    keys=set(a); assert keys==set(b)
    return dict(n=len(keys),retained=sum(a[k] and b[k] for k in keys),
        lost=sum(a[k] and not b[k] for k in keys),gained=sum(not a[k] and b[k] for k in keys),
        neither=sum(not a[k] and not b[k] for k in keys))

def dist_path(path,goal):
    p=np.asarray(path,float); g=np.asarray(goal,float)
    a,b=p[:-1],p[1:]; d=b-a; denom=(d*d).sum(1)
    t=np.divide(((g-a)*d).sum(1),denom,out=np.zeros(len(d)),where=denom>0).clip(0,1)
    return float(np.linalg.norm(a+t[:,None]*d-g,axis=1).min())

def supplement(report):
    rng=np.random.default_rng(20261026)
    intervals={}
    for a,b in [('C','T'),('u128','C'),('u128','T')]:
        groups=defaultdict(list)
        for row in report['task_deltas']:groups[row['world']].append((row[b]-row[a])/5.)
        values=np.array([np.mean(v) for v in groups.values()])
        boot=values[rng.integers(len(values),size=(20000,len(values)))].mean(1)*100
        intervals[a+'->'+b]=dict(worlds=len(values),delta_pp=float(values.mean()*100),
            descriptive_world_bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist(),
            positive_worlds=int((values>0).sum()),negative_worlds=int((values<0).sum()))
    report['cluster_intervals']=intervals
    old=PROJECT/'wsl_snapshot/routea_common_a1_v1_run01/training_b2560'
    oldm=read(next(old.rglob('metrics_update_000129.json')))
    cm=read(RETURN/'arms/C/metrics_update_000129.json'); replay=read(RETURN/'old_replay/metrics_update_000129.json')
    clean=json.loads(json.dumps(cm['updates']))
    for u in clean:u['pre_update'].pop('keep_diagnostics',None)
    assert clean==replay['updates']
    report['continuity_scope']=dict(C_vs_old_replay_update_audit_exact=True,
        C_vs_historical_uninterrupted_u129_exact=clean==oldm['updates'],
        historical_first_policy_loss=oldm['updates'][0]['pre_update']['policy_loss'],
        resumed_first_policy_loss=cm['updates'][0]['pre_update']['policy_loss'])
    boundary={}
    for arm in ['C','T']:
        d=read(RETURN/f'val_{arm}/arrival_boundary_audit.json'); excess=[]
        for e in d['accepted_boundary_events']:
            distance=float(np.linalg.norm(np.array(e['terminal_xyz'])-np.array(e['goal_xyz'])))
            assert abs(distance-e['distance_m'])<1e-12
            assert 0<distance-.5<=e['rounding_allowance_m'] and not e['terminal_safety_flags']
            excess.append(distance-.5)
        assert len(excess)==d['accepted_boundary_count']
        boundary[arm]=dict(events=len(excess),max_excess_m=max(excess))
    report['boundary_recomputed']=boundary
    oldval=PROJECT/'wsl_snapshot/routea_a1_mechanism_val20_run03/val_u128/paired_episodes.jsonl'
    if oldval.exists():
        oldrows=[json.loads(x) for x in oldval.read_text(encoding='utf-8').splitlines()]
        key=lambda r:(r['task_id'],r['seed'])
        parent={key(r):r['done_reason']=='reached' for r in oldrows if r['policy']=='rl'}
        for arm in ['C','T']:
            rows=[json.loads(x) for x in (RETURN/f'val_{arm}/paired_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            child={key(r):r['done_reason']=='reached' for r in rows if r['policy']=='rl'}
            report['val'][arm]['vs_u128']=pair(parent,child)
    report['execution_rc']={p.stem:int(p.read_text().strip()) for p in (RETURN/'execution').glob('*.rc')}

def main():
    report={}; bad=[]; lines=(RETURN/'return_sha256.txt').read_text().splitlines()
    for line in lines:
        expected,name=line.split(None,1); p=RETURN/name.lstrip('*')
        if not p.is_file() or sha(p)!=expected: bad.append(name)
    substantive=[p for p in bad if not p.endswith(':Zone.Identifier')]
    report['integrity']=dict(files=len(lines),matched=len(lines)-len(bad),bad=bad,substantive_bad=substantive)
    assert not substantive,substantive
    print('HASH',len(lines),flush=True)
    idx=read(RETURN/'evaluation_dev/paired_index.json')
    names={n:('BC' if n=='bc' else 'u128' if d['update']==128 else 'C' if '/C/' in d['checkpoint_path'] else 'T') for n,d in idx['models'].items()}
    records={v:{} for v in names.values()}; meta={}; hashes=defaultdict(dict)
    for entry in idx['records']:
        key=entry['episode_key']; assert set(entry['models'])==set(names)
        for name,item in entry['models'].items():
            label=names[name]; base=RETURN/'evaluation_dev/full'/name
            commit=read(base/item['commit']); trace=base/commit['trace']
            assert sha(trace)==commit['sha256']==item['trace_sha256']
            with trace.open(encoding='utf-8') as f: rows=[json.loads(line) for line in f]
            h,s=rows[0],rows[-1]; assert s==commit['summary']
            cycles=rows[1:-1]
            minimum=min(dist_path(c['executed_path'],h['task']['goal_pos']) for c in cycles)
            assert abs(minimum-s['trajectory_metrics']['min_goal_distance_m'])<1e-5
            assert len(cycles)==s['decisions']
            assert cycles[-1]['done_reason']==s['done_reason']
            assert h['use_base_policy']==(label=='BC')
            assert h['initial_hash']==s['initial_hash']
            assert key not in records[label]
            obs=[ObstacleSpec.from_dict(o) for o in h['geometry']]
            blocked=segment_obstacle_collision(np.array(h['task']['start_pos']),np.array(h['task']['goal_pos']),obs,.1) is not None
            task_meta=dict(task=h['task_id'],world=h['world_id'],role=h['role'],blocked=blocked,initial=h['initial_hash'])
            if key in meta: assert meta[key]==task_meta
            meta[key]=task_meta
            records[label][key]=s; hashes[label][key]=commit['sha256']
        if len(records['BC'])%100==0: print('TRACES',len(records['BC']),flush=True)
    assert all(len(v)==420 for v in records.values())
    success={label:{k:r['safe_arrival'] for k,r in rs.items()} for label,rs in records.items()}
    report['dev']={label:dict(outcomes=dict(Counter(r['done_reason'] for r in rs.values())),
        near1_timeouts=sum(r['done_reason']=='time_limit' and r['trajectory_metrics']['min_goal_distance_m']<=1 for r in rs.values()),
        roles={role:sum(r['safe_arrival'] for k,r in rs.items() if meta[k]['role']==role) for role in sorted({m['role'] for m in meta.values()})},
        blocked_dense=dict(n=sum(m['blocked'] and m['role']=='dense_recovery' for m in meta.values()),success=sum(r['safe_arrival'] for k,r in rs.items() if meta[k]['blocked'] and meta[k]['role']=='dense_recovery')))
        for label,rs in records.items()}
    report['pairs']={a+'->'+b:pair(success[a],success[b]) for a,b in [('BC','u128'),('BC','C'),('BC','T'),('u128','C'),('u128','T'),('C','T')]}
    strata={category:[k for k in meta if (success['BC'][k],success['u128'][k])==flags] for category,flags in [('both',(True,True)),('bc_only',(True,False)),('rl_only',(False,True)),('neither',(False,False))]}
    report['baseline_strata']={cat:dict(n=len(keys),C=sum(success['C'][k] for k in keys),T=sum(success['T'][k] for k in keys)) for cat,keys in strata.items()}
    report['task_deltas']=[]
    for task in sorted({m['task'] for m in meta.values()}):
        keys=[k for k in meta if meta[k]['task']==task]
        report['task_deltas'].append(dict(task=task,world=meta[keys[0]]['world'],role=meta[keys[0]]['role'],blocked=meta[keys[0]]['blocked'],**{label:sum(success[label][k] for k in keys) for label in records}))
    report['transition_C_T']=dict(Counter(records['C'][k]['done_reason']+'->'+records['T'][k]['done_reason'] for k in meta))
    report['training']={}
    for arm in ['C','T']:
        files=sorted((RETURN/'arms'/arm).glob('metrics_update_*.json')); assert len(files)==128
        ratios=[]; cosines=[]; losses=[]; commits=0; rejects=0; blocks=[]
        for u,p in enumerate(files,129):
            assert p.name==f'metrics_update_{u:06d}.json'
            m=read(p); assert len(m['updates'])==20
            for update in m['updates']:
                commits+=update['status']=='committed'; rejects+=update['status']=='rejected'
                d=update['pre_update'].get('keep_diagnostics')
                if arm=='T':
                    assert d and d['coefficient']==read(RETURN/'first_update_gate.json')['coefficient']
                    ratios.append(d['ratio']);cosines.append(d['cosine']);losses.append(d['keep_loss'])
                else: assert d is None
        def stats(x):return dict(min=float(np.min(x)),median=float(np.median(x)),max=float(np.max(x)),mean=float(np.mean(x))) if x else None
        report['training'][arm]=dict(updates=len(files),commits=commits,rejects=rejects,final_counters=m['counters'],ratio=stats(ratios),cosine=stats(cosines),keep_loss=stats(losses),negative_cosine_fraction=float(np.mean(np.array(cosines)<0)) if cosines else None,
            first160_ratio=stats(ratios[:32*20]),last32_ratio=stats(ratios[-32*20:]))
    vals={}
    for arm in ['C','T']:
        rows=[json.loads(x) for x in (RETURN/f'val_{arm}/paired_episodes.jsonl').read_text().splitlines()]; assert len(rows)==160
        vals[arm]={}
        for policy in ['bc','rl']:
            selected=[r for r in rows if r['policy']==policy]; assert len(selected)==80
            vals[arm][policy]={(r['task_id'],r['seed']):r for r in selected}; assert len(vals[arm][policy])==80
    assert vals['C']['bc']==vals['T']['bc']
    report['val']={arm:dict(outcomes=dict(Counter(r['done_reason'] for r in vals[arm]['rl'].values())),vs_BC=pair({k:r['done_reason']=='reached' for k,r in vals[arm]['bc'].items()},{k:r['done_reason']=='reached' for k,r in vals[arm]['rl'].items()})) for arm in vals}
    report['val']['C->T']=pair({k:r['done_reason']=='reached' for k,r in vals['C']['rl'].items()},{k:r['done_reason']=='reached' for k,r in vals['T']['rl'].items()})
    report['val']['BC']=dict(Counter(r['done_reason'] for r in vals['C']['bc'].values()))
    report['gate']=read(RETURN/'first_update_gate.json'); report['calibration']=read(RETURN/'arms/T/keep_calibration.json')
    inv=read(RETURN/'reference/inventory.json');report['reference']={k:v for k,v in inv.items() if k not in ('episodes','categories')}
    report['completion']=read(RETURN/'evaluation_dev/run_complete_006.json')
    report['dev_trace_hashes']=dict(hashes)
    supplement(report)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['dev','pairs','baseline_strata','training','val']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
