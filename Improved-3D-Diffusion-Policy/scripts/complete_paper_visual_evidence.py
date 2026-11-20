"""Render real archived execution trajectories; never run a simulator."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from export_paper_trajectory_metrics import measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True)
    parser.add_argument('--gazebo',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--failure-traces')
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(exist_ok=True)
    z=zipfile.ZipFile(args.package)
    raw=lambda s:z.read(next(i for i in z.infolist() if i.filename.endswith(s)))
    selected=json.loads(raw('selected_trajectories.json'))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['pdf','svg','png']: fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig=plt.figure(figsize=(12,7.2)); metrics=[]
    for col,category in enumerate(['稀疏','适中','密集']):
        ax=fig.add_subplot(2,3,col+1,projection='3d'); curve=fig.add_subplot(2,3,col+4)
        group=[x for x in selected if x['selection']['category']==category]
        for item in group:
            trace=out/(item['policy']+'_'+category+'_source.jsonl')
            lines=[item['header']]+item['transitions']+[item['end']]
            trace.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in lines)+'\n',encoding='utf-8')
            m=measure(trace)
            m.update(upstream_source=item['source'],upstream_sha256=item['sha256'],
                     registered_policy=item['policy'],input_scope='Selected archived trajectory from uploaded paper package; reconstructed JSONL retains original transition objects.')
            metrics.append(m)
            xyz=np.asarray(m['executed_xyz']);goal=np.asarray(item['header']['task']['goal_pos'])
            label='UAV-iDP3' if item['policy']=='P0' else '本文方法'
            color='#7C8796' if item['policy']=='P0' else '#0072B2'
            ax.plot(*xyz.T,label=label,color=color,ls='--' if item['policy']=='P0' else '-')
            ax.scatter(*xyz[0],color=color,marker='s',s=15)
            ax.scatter(*xyz[-1],color=color,marker='o' if m['paper_safe_arrival'] else 'x',s=25)
            distance=np.linalg.norm(xyz-goal,axis=1)
            length=np.r_[0,np.linalg.norm(np.diff(xyz,axis=0),axis=1).cumsum()]
            curve.plot(length,distance,color=color,label=label,ls='--' if item['policy']=='P0' else '-')
        ax.scatter(*goal,color='#D99036',marker='*',s=55)
        for o in group[0]['header']['geometry']:
            h=o['height'] if o['shape']=='cylinder' else o['size_z']
            if o['shape']=='cylinder':
                theta=np.linspace(0,2*np.pi,24);zz=np.array([o['z']-h/2,o['z']+h/2])
                ax.plot_surface(o['x']+o['radius']*np.cos(theta)[None,:]+np.zeros((2,1)),
                  o['y']+o['radius']*np.sin(theta)[None,:]+np.zeros((2,1)),zz[:,None]+np.zeros((1,24)),color='#AAAAAA',alpha=.18)
            else:
                xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array([o['size_x'],o['size_y']])/2
                c,s=np.cos(o['yaw']),np.sin(o['yaw']);xy=xy@np.array([[c,s],[-s,c]])+[o['x'],o['y']]
                v=np.vstack([np.c_[xy,np.full(4,o['z']-h/2)],np.c_[xy,np.full(4,o['z']+h/2)]])
                ax.add_collection3d(Poly3DCollection([v[idx] for idx in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]],facecolor='#AAAAAA',alpha=.15))
        limits=np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]);ranges=np.ptp(limits,axis=1)
        ax.set_box_aspect(ranges);ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)');ax.set_title(category);ax.legend(fontsize=8)
        curve.axhline(1.5,color='#999999',ls=':');curve.set_xlabel('累计执行路长（m）');curve.set_ylabel('目标距离（m）');curve.legend(fontsize=8)
    save(fig,'图6_真实三维轨迹与目标接近')
    (out/'selected_trajectory_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.failure_traces:
        eligible={r['任务ID'] for r in csv.DictReader(io.StringIO(raw('kinematic_episodes.csv').decode('utf-8-sig'))) if r['计入正文统计']=='是'}
        for path in sorted(Path(args.failure_traces).glob('*.jsonl')):
            with path.open('rb') as f:
                header=json.loads(f.readline());f.seek(max(0,path.stat().st_size-16384));tail=f.read().splitlines()
            end=json.loads(tail[-1])
            if header['task_id'] in eligible and end.get('done_reason')=='collision':
                failure=measure(path)
                if failure['paper_safe_arrival']: continue
                xyz=np.asarray(failure['executed_xyz']);goal=np.asarray(header['task']['goal_pos'])
                fig=plt.figure(figsize=(7,3.5));ax=fig.add_subplot(121,projection='3d');curve=fig.add_subplot(122)
                ax.plot(*xyz.T,color='#0072B2');ax.scatter(*xyz[-1],marker='x',color='#CA7979');ax.scatter(*goal,marker='*',color='#D99036')
                ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
                curve.plot(np.r_[0,np.linalg.norm(np.diff(xyz,axis=0),axis=1).cumsum()],np.linalg.norm(xyz-goal,axis=1))
                curve.set_xlabel('累计执行路长（m）');curve.set_ylabel('目标距离（m）');curve.axhline(1.5,ls=':',color='#999999')
                save(fig,'附图_本文方法真实失败案例')
                (out/'failure_case.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf-8')
                break
    # Match the actual 40-task list and use only a common recorded trial.
    flights=list(csv.DictReader(io.StringIO(raw('gazebo_flights.csv').decode('utf-8-sig'))))
    ids=sorted({r['任务ID前12位'] for r in flights if r['计入正文统计']=='是'})
    root=Path(args.gazebo)
    task=next(t for t in ids if all((root/p/(t+'_r1')/'result_state.csv').exists() for p in ['P0','P6','EGO']))
    fig=plt.figure(figsize=(9,4));ax=fig.add_subplot(121,projection='3d');err=fig.add_subplot(122)
    provenance=[]
    for policy,label,color in [('P0','UAV-iDP3','#7C8796'),('P6','本文方法','#0072B2'),('EGO','EGO-Planner','#D99036')]:
        folder=root/policy/(task+'_r1')
        state=list(csv.DictReader((folder/'result_state.csv').open(encoding='utf-8-sig')))
        t=np.array([float(r['timestamp']) for r in state]);xyz=np.array([[float(r[k]) for k in ['px','py','pz']] for r in state])
        ax.plot(*xyz.T,label=label,color=color)
        cp=folder/'result_pos_cmd.csv';commands=list(csv.DictReader(cp.open(encoding='utf-8-sig')))
        ct=np.array([float(r['timestamp']) for r in commands]);ref=np.array([[float(r[k]) for k in ['cmd_px','cmd_py','cmd_pz']] for r in commands])
        mask=(t>=ct[0])&(t<=ct[-1]);ti=t[mask]
        # Piecewise-linear interpolation only over overlapping timestamps.
        reference=np.column_stack([np.interp(ti,ct,ref[:,i]) for i in range(3)])
        e=np.linalg.norm(xyz[mask]-reference,axis=1)
        err.plot(ti-ti[0],e,label=label,color=color)
        provenance.append(dict(policy=policy,task=task,trial=1,state=str(folder/'result_state.csv'),
           reference=str(cp),effective_samples=len(e),rmse_m=float(np.sqrt(np.mean(e*e))),
           alignment='Linear interpolation of reference position at state timestamps within overlapping interval; descriptive single-trial tracking discrepancy, not controller latency compensated error.'))
    ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)');ax.legend(fontsize=8)
    err.set_xlabel('重叠记录起点后的时间（s）');err.set_ylabel('参考与实际位置差（m）');err.legend(fontsize=8)
    save(fig,'图9_Gazebo实际轨迹与位置差')
    (out/'gazebo_figure_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rendered six archived kinematic trajectories and a common Gazebo trial',task)


if __name__=='__main__': main()
