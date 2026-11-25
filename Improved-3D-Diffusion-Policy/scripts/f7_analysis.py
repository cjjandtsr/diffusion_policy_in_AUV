"""F7 matched collection, non-stopping monitor and paired E2 endpoint analysis."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import read, write, digest


def verdict(arrival, lower, collision, seed_deltas, complete=True):
    if not complete:return '无结论'
    if lower>0 and collision<0 and all(x>0 for x in seed_deltas):return '符合采纳标准'
    if arrival<=0 or collision>=0:return '本预算下不符合采纳标准'
    return '不确定'


def matched_rollout(prepared, control):
    from omegaconf import OmegaConf
    from scripts.f6_u129_matched_rollout import IDENT, _identical
    t={'P18':'P16','P19':'P17'}[control]
    spec=read(Path(prepared)/'prepared.json');configs={}
    data={}
    for name in (t,control):
        item=spec['configs'][name];assert digest(item['path'])==item['sha256']
        c=OmegaConf.load(item['path']);configs[name]=c
        m=read(Path(c.result_dir)/'metrics_update_000257.json')
        assert len(m['updates'])==20 and all(u['status']=='committed' for u in m['updates'])
        rows=m['rollout_audit']['transition_records']
        data[name]={r['transition_id']:r for r in rows}
        assert len(rows)==len(data[name])==1024
    assert data[t].keys()==data[control].keys()
    for tid,a in data[t].items():
        b=data[control][tid]
        for key in IDENT+('executed_path','f7_audit'):
            assert key in a and key in b and _identical(a[key],b[key]),(tid,key)
        assert a['update_index']==257 and a['policy_state_update_index']==256
        assert set(a['reward_terms'])==set(b['reward_terms'])|{'approach_cbf'}
        for key,value in b['reward_terms'].items():assert _identical(value,a['reward_terms'][key]),(tid,key)
        d=a['f7_audit']
        c=d['point_clearances'];dt=d['segment_durations']
        expected=-float(configs[t].f7.approach_weight)*sum(seconds*min(2.,max(0.,-((c[i+1]-c[i])/seconds+(c[i+1]+c[i]-.6)/6.)))
            for i,seconds in enumerate(dt) if seconds>0.)
        assert abs(a['reward_terms']['approach_cbf']-expected)<1e-9
        assert abs(a['reward']-b['reward']-expected)<1e-9
    return dict(passed=True,pair=[t,control],transitions=1024,update=257,
                claim='matched collection only; updated tensors need not match')


def monitor_training(directory, update):
    directory=Path(directory);stats=[]
    # Read only lightweight summaries from the most recent eight and first eight updates.
    needed=sorted(set(range(257,min(update,264)+1))|set(range(max(257,update-7),update+1)))
    for u in needed:
        m=read(directory/('metrics_update_%06d.json'%u))
        rows=m['rollout_audit']['transition_records']
        reasons=[r['terminal_facts']['done_reason'] for r in rows if r['terminated'] or r['truncated']]
        # collection_boundary alone is not a completed episode.
        reasons=[x for x in reasons if x!='none']
        speeds=[];lateral=[];speed_changes=[]
        for i,r in enumerate(rows):
            path=np.asarray(r['executed_path']);dt=np.asarray(r['f7_audit']['segment_durations'])
            lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
            speeds.extend((lengths[dt>0]/dt[dt>0]).tolist())
            if i+1<len(rows) and r['episode_id']==rows[i+1]['episode_id'] and r['f7_audit']['approach_cbf_diagnostic']!=0:
                nxt=rows[i+1];p=np.asarray(nxt['executed_path']);d=path[-1]-path[0];delta=p[-1]-p[0];norm=np.linalg.norm(d)
                lateral.append(float(np.linalg.norm(delta) if norm==0 else np.linalg.norm(delta-d*np.dot(delta,d)/norm**2)))
                nt=sum(nxt['f7_audit']['segment_durations'])
                if nt>0 and dt.sum()>0:speed_changes.append(float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()/nt-lengths.sum()/dt.sum()))
        pre=[x['pre_update'] for x in m['updates'] if x['status']=='committed']
        stats.append(dict(update=u,completed=len(reasons),reached=reasons.count('reached'),
            rate=reasons.count('reached')/len(reasons) if reasons else None,
            outcomes=dict(Counter(reasons)),lr=m['formal_training']['f7_actor_lr'],
            clip_fraction=distribution(x['clip_fraction'] for x in pre),
            approx_kl=distribution(x['approx_kl'] for x in pre),
            ratio_min=min(x['slice_diagnostics']['full']['ratio_min'] for x in pre),
            ratio_max=max(x['slice_diagnostics']['full']['ratio_max'] for x in pre),
            actual_submission_example=pre[0]['slice_diagnostics']['full']['f7_example'],
            approach_cbf=distribution(r['f7_audit']['approach_cbf_diagnostic'] for r in rows),
            approach_nonzero_fraction=sum(r['f7_audit']['approach_cbf_diagnostic']!=0 for r in rows)/len(rows),
            speeds=distribution(speeds),next_lateral_displacements=distribution(lateral),
            next_speed_changes=distribution(speed_changes)))
    baseline=[s['rate'] for s in stats if s['update']<=264 and s['rate'] is not None]
    recent=[s['rate'] for s in stats if s['update']>=update-7 and s['rate'] is not None]
    alarm=update>264 and len(baseline)==8 and len(recent)==8 and np.mean(recent)<np.mean(baseline)/2
    write(directory/('f7_monitor_%06d.json'%update),dict(updates=stats,arrival_alarm=bool(alarm),
          automatic_stop=False,rate_definition='mean of eight per-update completed-episode arrival rates'))


def paired_stats(groups):
    # groups: world -> all matched condition/seed differences in that world.
    names=sorted(groups);arrays=[np.asarray(groups[w],float) for w in names]
    sums=np.array([a.sum(axis=0) for a in arrays]);counts=np.array([len(a) for a in arrays])
    rng=np.random.default_rng(2026102601);draw=rng.integers(0,len(names),(20000,len(names)))
    samples=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)[:,None]
    point=sums.sum(axis=0)/counts.sum()
    return dict(arrival=float(point[0]),collision=float(point[1]),
                arrival_ci=np.quantile(samples[:,0],[.025,.975]).tolist(),worlds=len(names),
                paired_records=int(counts.sum()),bootstrap=20000,seed=2026102601)


def decision_geometry(path, durations, obstacles, weight):
    from diffusion_policy_3d.env.reward_v5 import approach_cost
    from diffusion_policy_3d.env.reward_v2_support.reward import _obstacle_clearances
    path=np.asarray(path,float);dt=np.asarray(durations,float)
    lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
    speeds=np.divide(lengths,dt,out=np.zeros_like(dt),where=dt>0)
    cost,clearances=approach_cost(path,obstacles,dt,weight)
    gaps=_obstacle_clearances(path,obstacles)
    return dict(approach_cbf_diagnostic=cost,speeds=speeds.tolist(),
                near_speeds=speeds[gaps<2.5].tolist(),min_clearance=float(gaps.min()),
                length=float(lengths.sum()),elapsed=float(dt.sum()))


def outcome_1m(rows, goal, raw_outcome, radius=1.):
    # Paper arrival_1m.py: raw reached OR continuous executed-path nearest < 1.
    # The executed path already ends at the first physical failure.
    from scripts.run_routea_xk_evaluation import closest_path_point
    path=np.concatenate([np.asarray(r['executed_path'],float) for r in rows])
    nearest=float(np.linalg.norm(closest_path_point(path,goal)-goal))
    if raw_outcome=='reached' or nearest<radius:return 'reached'
    return 'collision' if raw_outcome in ('collision','ground_crash','crash_ground') else 'time_limit'


def first_strict_entry(path, goal):
    from scripts.run_routea_xk_evaluation import closest_path_point
    from diffusion_policy_3d.env.reward_v2_support.first_event import _ball_entry
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        if np.linalg.norm(closest_path_point(np.stack((a,b)),goal)-goal)<1.:
            return i,_ball_entry(a,b,goal,1.)
    return None


def read_evaluation(job, bank, weight, identity):
    from scripts.routea_paired_support.episodes import load_commit
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision
    from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
    from scripts.run_routea_xk_evaluation import closest_path_point
    job=Path(job);records={};expected={e['key']:e for e in bank['episodes']}
    tasks={t['task_id']:t for t in bank['tasks']}
    assert len(expected)==len(bank['episodes']), 'duplicate bank condition'
    assert len({(e['task_id'],e['repeat']) for e in bank['episodes']})==len(expected)
    assert {p.stem for p in (job/'commits').glob('*.json')}==set(expected),'missing or extra conditions'
    for key,e in expected.items():
        h,rows,end=load_commit(job,job/'commits'/(key+'.json'))
        task=tasks[e['task_id']];world=bank['worlds'][task['world_id']]
        assert h['task_id']==e['task_id'] and h['repeat']==e['repeat']
        assert h['task']==task['payload'] and h['role']==task['role'] and h['world_id']==task['world_id']
        assert h['geometry']==world['geometry'] and h['obstacle_sha256']==world['obstacle_sha256']
        assert h['bank_hash']==bank['content_hash'] and h['initial_hash']==e['initial_hash']
        assert not h['use_base_policy']
        assert h['policy_identity_hash']==identity
        assert h['terminate_at_space_limits'] is False
        obs=[ObstacleSpec(**o) for o in h['geometry']];goal=np.asarray(h['task']['goal_pos'],np.float32)
        outcome=outcome_1m(rows,goal,end['done_reason']);first=None;elapsed=length=0.;details=[]
        for r in rows:
            path=np.asarray(r['executed_path'],np.float32)
            dt=np.full(len(path)-1,.3)
            if r['event_fraction'] is not None:dt[-1]*=r['event_fraction']
            d=decision_geometry(path,dt,obs,weight);details.append(d)
            if first is None:
                entry=first_strict_entry(path,goal)
                if entry is not None and outcome=='reached':
                    first=r['decision'];i,f=entry
                    elapsed+=float(dt[:i].sum()+dt[i]*f)
                    length+=float(np.linalg.norm(np.diff(path[:i+1],axis=0),axis=1).sum()+np.linalg.norm(path[i+1]-path[i])*f)
                else:elapsed+=d['elapsed'];length+=d['length']
        lateral=[];speed_changes=[]
        for i,d in enumerate(details[:-1]):
            if d['approach_cbf_diagnostic']==0:continue
            a=np.asarray(rows[i]['executed_path']);b=np.asarray(rows[i+1]['executed_path'])
            direction=a[-1]-a[0];norm=np.linalg.norm(direction)
            delta=b[-1]-b[0]
            lateral.append(float(np.linalg.norm(delta) if norm==0 else np.linalg.norm(delta-direction*np.dot(delta,direction)/norm**2)))
            speed_changes.append(details[i+1]['length']/max(details[i+1]['elapsed'],1e-12)-d['length']/max(d['elapsed'],1e-12))
        early=None
        if end['done_reason']=='collision' and len(rows)>=3:
            hit=locate_first_event(np.asarray(rows[-1]['candidate_h4_path']),goal,obs,.1,5.,terminate_at_space_limits=False)
            if hit is not None and hit.kind=='collision':
                target=[o for o in obs if o.name==hit.obstacle_name]
                plan=np.asarray(rows[-3]['planned_policy_path'])
                early=any(segment_obstacle_collision(a,b,target,.1) is not None for a,b in zip(plan[:-1],plan[1:]))
        records[(e['task_id'],e['repeat'])]=dict(world=h['world_id'],role=h['role'],
            initial_hash=h['initial_hash'],outcome=outcome,raw_outcome=end['done_reason'],
            outcome_1p5m_descriptive=outcome_1m(rows,goal,end['done_reason'],radius=1.5),
            success_time_s=elapsed if outcome=='reached' else None,
            path_efficiency=(max(0.,float(np.linalg.norm(np.asarray(h['task']['start_pos'])-goal))-1.)/length if outcome=='reached' and length else None),
            min_clearance=min(d['min_clearance'] for d in details),
            near_goal_timeout=bool(outcome=='time_limit' and np.linalg.norm(closest_path_point(np.concatenate([r['executed_path'] for r in rows]),goal)-goal)<2.),
            speeds=[v for d in details for v in d['speeds']],near_speeds=[v for d in details for v in d['near_speeds']],
            next_lateral_displacements=lateral,next_speed_changes=speed_changes,
            collision_plan_two_decisions_before=early,decisions=details)
    return records


def distribution(values):
    a=np.asarray([x for x in values if x is not None],float)
    assert np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        quantiles=np.quantile(a,[0,.1,.5,.9,1.]).tolist() if len(a) else [])


def describe(records, strata):
    rows=list(records.values());layers={}
    for field in ('w1','role'):
        for label in sorted({strata[k[0]]['labels']['w1'] if field=='w1' else v['role'] for k,v in records.items()}):
            selected=[v for k,v in records.items() if (strata[k[0]]['labels']['w1'] if field=='w1' else v['role'])==label]
            layers[field+':'+label]=dict(n=len(selected),outcomes=dict(Counter(v['outcome'] for v in selected)),
                collision_rate=sum(v['outcome']=='collision' for v in selected)/len(selected))
    return dict(n=len(rows),outcomes=dict(Counter(r['outcome'] for r in rows)),
        outcomes_1p5m_descriptive=dict(Counter(r['outcome_1p5m_descriptive'] for r in rows)),
        raw_half_meter_outcomes=dict(Counter(r['raw_outcome'] for r in rows)),layers=layers,
        success_time_s=distribution(r['success_time_s'] for r in rows),
        path_efficiency=distribution(r['path_efficiency'] for r in rows),
        minimum_clearance=distribution(r['min_clearance'] for r in rows),
        near_goal_timeouts=sum(r['near_goal_timeout'] for r in rows),
        near_miss_successes=sum(r['min_clearance']<.3 and r['outcome']=='reached' for r in rows),
        near_miss_count=sum(r['min_clearance']<.3 for r in rows),
        near_miss_arrival_rate=(sum(r['min_clearance']<.3 and r['outcome']=='reached' for r in rows)/sum(r['min_clearance']<.3 for r in rows) if any(r['min_clearance']<.3 for r in rows) else None),
        collision_plan_two_decisions_before=dict(Counter(str(r['collision_plan_two_decisions_before']) for r in rows if r['raw_outcome']=='collision')),
        **{k:distribution(x for r in rows for x in r[k]) for k in
           ('speeds','near_speeds','next_lateral_displacements','next_speed_changes')})


def evaluate(args):
    from scripts.routea_paired_support.core import canonical_hash
    from scripts.analyze_routea_f5 import check_bank
    from omegaconf import OmegaConf
    bank=read(args.bank)
    check_bank(bank,'dev420' if args.conditions==420 else 'val20')
    hashes={420:'863bf1fb2a41c991b73cc92493aa7d8732c711b4e48c6a8d063bd2dd8c59996a',
            80:'a2b28e0cf2ca32ee8a987d0f1a03f94d2783dea57fee1bd096fdb8e5ca542017'}
    assert bank['content_hash']==hashes[args.conditions]
    assert len(bank['episodes'])==args.conditions
    frozen=read(args.strata)
    assert frozen['bank_hash']==bank['content_hash'] and canonical_hash({k:v for k,v in frozen.items() if k!='content_hash'})==frozen['content_hash']
    strata={t['task_id']:t for t in frozen['tasks']}
    assert set(strata)=={t['task_id'] for t in bank['tasks']}
    prepared=read(Path(args.prepared)/'prepared.json')
    identities={}
    for name,item in prepared['configs'].items():
        assert digest(item['path'])==item['sha256']
        c=OmegaConf.load(item['path']);checkpoint=Path(c.output_dir)/'checkpoint_update_000320'
        h=read(checkpoint/'header.json');p=read(Path(args.jobs)/name/'policy_provenance.json')
        assert h['update_index']==320 and p['checkpoint_state_sha256']==digest(checkpoint/'state.pt')
        identities[name]=h['runtime_identity_hash']
        assert p['training_runtime_identity']==h['runtime_identity_hash'] and not p['actual_use_base_policy']
        assert p['policy']==name and p['bank_hash']==bank['content_hash']
        protocol=p['evaluation_sampling_protocol']
        assert protocol['sampling_mode']=='deterministic' and protocol['clip_protocol']['override']==2.
        for field in ('min_sampling_denoising_std','min_logprob_denoising_std'):
            assert np.allclose(np.asarray(protocol[field]),.1,atol=1e-8,rtol=0.)
    records={n:read_evaluation(Path(args.jobs)/n,bank,args.weight,identities[n]) for n in ('P16','P17','P18','P19')}
    records['P6']=read_evaluation(args.parent_job,bank,args.weight,'d987235a0cbb5ad395bbd6e7979bdb8d7e0cff1303b0611cb462c7d554bbd61f')
    groups=defaultdict(list);seed_deltas=[];matrices={}
    for a,b in (('P16','P18'),('P17','P19')):
        diffs=[];matrix=Counter()
        assert records[a].keys()==records[b].keys()
        for key,x in records[a].items():
            y=records[b][key];assert x['initial_hash']==y['initial_hash'] and x['world']==y['world']
            diff=[int(x['outcome']=='reached')-int(y['outcome']=='reached'),int(x['outcome']=='collision')-int(y['outcome']=='collision')]
            diffs.append(diff[0]);groups[x['world']].append(diff);matrix[y['outcome']+' -> '+x['outcome']]+=1
        seed_deltas.append(float(np.mean(diffs)));matrices[a+'-'+b]=dict(matrix)
    stats=paired_stats(groups)
    report=dict(primary_outcome_radius_m=1.,descriptive_outcome_radius_m=1.5,
        stats=stats,seed_arrival_deltas=seed_deltas,paired_transition_matrices=matrices,
        verdict=verdict(stats['arrival'],stats['arrival_ci'][0],stats['collision'],seed_deltas) if args.conditions==420 else '仅描述',
        control_arrival_difference=float(np.mean([int(records['P18'][k]['outcome']=='reached')-int(records['P19'][k]['outcome']=='reached') for k in records['P18']])),
        descriptions={n:describe(d,strata) for n,d in records.items()},
        versus_P6={n:{outcome:sum(int(v['outcome']==outcome)-int(records['P6'][k]['outcome']==outcome) for k,v in d.items()) for outcome in ('reached','collision','time_limit')} for n,d in records.items() if n!='P6'},
        bank_hash=bank['content_hash'],strata_hash=frozen['content_hash'],
        path_efficiency_definition='(initial distance to goal center - 1m) / path length until first 1m entry',
        records={n:[dict(task_id=k[0],repeat=k[1],**v) for k,v in d.items()] for n,d in records.items()})
    write(args.output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('matched');a.add_argument('--prepared',required=True);a.add_argument('--control',choices=('P18','P19'),required=True);a.add_argument('--output',required=True)
    a=s.add_parser('evaluate')
    for k in ('jobs','bank','output','strata','parent-job','prepared'):a.add_argument('--'+k,required=True)
    a.add_argument('--weight',type=float,choices=(1.,2.),required=True)
    a.add_argument('--conditions',type=int,choices=(80,420),required=True)
    args=p.parse_args()
    if args.command=='matched':write(args.output,matched_rollout(args.prepared,args.control))
    else:
        try:evaluate(args)
        except (AssertionError,ValueError,FileNotFoundError) as exc:
            if not Path(args.output).exists():write(args.output,dict(verdict='无结论',reason=str(exc)))
            raise
