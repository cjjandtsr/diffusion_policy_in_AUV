"""Select a common successful Gazebo trial and plot unchanged paths + verified obstacles."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution(folder):
    state=list(csv.DictReader((folder/'result_state.csv').open(encoding='utf-8-sig')))
    cmd=list(csv.DictReader((folder/'result_pos_cmd.csv').open(encoding='utf-8-sig')))
    condition=json.loads((folder/'condition.json').read_text(encoding='utf-8-sig'))
    t=np.array([float(r['timestamp']) for r in state])
    xyz=np.array([[float(r[k]) for k in ('px','py','pz')] for r in state])
    # run_task_flight.sh: goal_odom.xy = goal_world.xy - spawn.xy; z unchanged.
    assert float(condition['env']['SPAWN_Z'])==0
    xyz[:,:2]+=np.asarray(condition['start_pos'])[:2]
    # Plot only navigation after the first logged command, excluding takeoff/landing.
    mask=t>=float(cmd[0]['timestamp']);t=t[mask];xyz=xyz[mask]
    goal=np.asarray(condition['goal_pos'])
    for i,(a,b) in enumerate(zip(xyz[:-1],xyz[1:])):
        delta=b-a;offset=a-goal
        if np.linalg.norm(offset)<=1.5:xyz=xyz[:i+1];t=t[:i+1];break
        aa=delta@delta;bb=2*(offset@delta);cc=offset@offset-1.5**2
        disc=bb*bb-4*aa*cc
        if aa>0 and disc>=0:
            root=(-bb-np.sqrt(disc))/(2*aa)
            if 0<=root<=1:
                xyz=np.vstack((xyz[:i+1],a+root*delta));t=np.r_[t[:i+1],t[i]+root*(t[i+1]-t[i])];break
    length=np.r_[0,np.linalg.norm(np.diff(xyz,axis=0),axis=1).cumsum()]
    # Resampling is solely a candidate-ranking measurement, never the plotted path.
    keep=np.r_[True,np.diff(length)>1e-9]
    grid=np.arange(0,length[-1],.25)
    sampled=np.column_stack([np.interp(grid,length[keep],xyz[keep,j]) for j in range(3)])
    direction=np.diff(sampled,axis=0);direction/=np.linalg.norm(direction,axis=1)[:,None]
    angles=np.arccos(np.clip(np.sum(direction[1:]*direction[:-1],axis=1),-1,1))
    smoothness=float(angles.sum()/length[-1])
    return xyz,t,condition,dict(turn_radians_per_m=smoothness,path_length_m=float(length[-1]),
             path_to_chord=float(length[-1]/np.linalg.norm(xyz[-1]-xyz[0])),samples=len(xyz))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','gazebo','geometry-index','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--rank',type=int,default=1)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    z=zipfile.ZipFile(a.package)
    raw=z.read(next(i for i in z.infolist() if i.filename.endswith('gazebo_flights.csv')))
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))));groups=defaultdict(dict)
    for r in rows:
        if r['计入正文统计']=='是' and r['监控终止状态']=='reached' and r['结局_1.5m']=='成功':
            groups[r['任务ID前12位'],int(r['重复'])][r['方法']]=r
    geometry_index=defaultdict(list)
    for line in Path(a.geometry_index).read_text(encoding='utf-8-sig').splitlines():
        geometry_index[Path(line).stem].append(Path(line))
    candidates=[];data={};root=Path(a.gazebo)
    for (task,repeat),methods in sorted(groups.items()):
        if set(methods)!={'P0','P6','EGO'}:continue
        runs={policy:execution(root/policy/(task+'_r'+str(repeat))) for policy in ['P0','P6','EGO']}
        conditions=[v[2] for v in runs.values()]
        identities={(c['world_id'],c['obstacle_sha256'],tuple(c['start_pos']),tuple(c['goal_pos'])) for c in conditions}
        assert len(identities)==1
        condition=conditions[0]
        assert all(not any(c.get('spawn_offset',[0,0,0])) for c in conditions), 'Nonzero obstacle transform requires explicit mapping'
        matched=[path for path in geometry_index[condition['world_id']] if sha(path)==condition['obstacle_sha256']]
        if not matched:continue
        geometry=matched[0]
        score=max(v[3]['turn_radians_per_m'] for v in runs.values())
        row=dict(task=task,repeat=repeat,world=condition['world_id'],worst_turn_radians_per_m=score,
                 mean_turn_radians_per_m=float(np.mean([v[3]['turn_radians_per_m'] for v in runs.values()])),geometry=str(geometry))
        for policy,v in runs.items():row[policy+'_turn_radians_per_m']=v[3]['turn_radians_per_m'];row[policy+'_path_length_m']=v[3]['path_length_m']
        candidates.append(row);data[task,repeat]=runs
    candidates.sort(key=lambda r:(r['worst_turn_radians_per_m'],r['mean_turn_radians_per_m'],r['task'],r['repeat']))
    with (out/'候选排序.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
    chosen=candidates[a.rank-1];runs=data[chosen['task'],chosen['repeat']]
    obstacles=json.loads(Path(chosen['geometry']).read_text(encoding='utf-8-sig'))['obstacles']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
        'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(10.8,4.8));ax=fig.add_subplot(121,projection='3d');top=fig.add_subplot(122)
    for o in obstacles:
        if o['shape']=='cylinder':
            theta=np.linspace(0,2*np.pi,40);zz=np.array([o['z']-o['height']/2,o['z']+o['height']/2])
            x=o['x']+o['radius']*np.cos(theta);y=o['y']+o['radius']*np.sin(theta)
            ax.plot_surface(np.tile(x,(2,1)),np.tile(y,(2,1)),np.tile(zz[:,None],(1,len(theta))),color='#9AA5AE',alpha=.24,shade=False)
            top.add_patch(Circle((o['x'],o['y']),o['radius'],facecolor='#D9DEE2',edgecolor='#9AA5AE',linewidth=.6,zorder=1))
        else:
            xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*[o['size_x']/2,o['size_y']/2]
            c,s=np.cos(o.get('yaw',0)),np.sin(o.get('yaw',0));xy=xy@np.array([[c,s],[-s,c]])+[o['x'],o['y']]
            v=np.vstack([np.c_[xy,np.full(4,o['z']-o['size_z']/2)],np.c_[xy,np.full(4,o['z']+o['size_z']/2)]])
            ax.add_collection3d(Poly3DCollection([v[idx] for idx in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]],facecolor='#AEB8C0',edgecolor='#919CA6',linewidth=.3,alpha=.22))
            top.add_patch(Polygon(xy,facecolor='#D9DEE2',edgecolor='#9AA5AE',linewidth=.6,zorder=1))
    sources=[]
    for policy,label,color,style in [('P0','UAV-iDP3','#0072B2','--'),('P6','本文方法','#D55E00','-'),('EGO','EGO-Planner','#009E73','-.')]:
        xyz,t,condition,metrics=runs[policy]
        ax.plot(*xyz.T,color=color,lw=2.1,ls=style,label=label,zorder=10)
        top.plot(*xyz[:,:2].T,color=color,lw=2.1,ls=style,label=label,zorder=3)
        ax.scatter(*xyz[-1],s=16,color=color,edgecolor='white',linewidth=.4,zorder=12)
        top.scatter(*xyz[-1,:2],s=16,color=color,edgecolor='white',linewidth=.4,zorder=4)
        folder=root/policy/(chosen['task']+'_r'+str(chosen['repeat']))
        table=out/(policy+'_displayed_raw_navigation.csv')
        np.savetxt(table,np.c_[t,xyz],delimiter=',',header='timestamp,x,y,z',comments='')
        sources.append(dict(policy=policy,condition=str(folder/'condition.json'),
            state=str(folder/'result_state.csv'),state_sha256=sha(folder/'result_state.csv'),
            result=str(folder/'result.json'),metrics=metrics,displayed_samples=str(table)))
    condition=runs['P0'][2];start=np.array(condition['start_pos']);goal=np.array(condition['goal_pos'])
    for xyz,marker,label,color in [(start,'s','起点','#333333'),(goal,'*','目标','#D6A83A')]:
        ax.scatter(*xyz,s=45 if marker=='s' else 95,color=color,marker=marker,edgecolor='white',linewidth=.7,zorder=12)
        top.scatter(*xyz[:2],s=45 if marker=='s' else 95,color=color,marker=marker,edgecolor='white',linewidth=.7,label=label,zorder=5)
    top.add_patch(Circle(goal[:2],1.5,fill=False,color='#B79C65',ls=':',lw=1.1,zorder=2))
    xs=[start[0],goal[0]-1.5,goal[0]+1.5];ys=[start[1],goal[1]-1.5,goal[1]+1.5];zs=[0,start[2],goal[2]]
    for o in obstacles:
        rx=o.get('radius',o.get('size_x',0)/2);ry=o.get('radius',o.get('size_y',0)/2)
        xs.extend([o['x']-rx,o['x']+rx]);ys.extend([o['y']-ry,o['y']+ry]);zs.append(o['z']+(o.get('height',o.get('size_z',0)))/2)
    for xyz,_,_,_ in runs.values():xs.extend(xyz[:,0]);ys.extend(xyz[:,1]);zs.extend(xyz[:,2])
    xl=(min(xs)-.8,max(xs)+.8);yl=(min(ys)-.8,max(ys)+.8);zl=(0,max(zs)+.4)
    ax.set(xlim=xl,ylim=yl,zlim=zl,xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    ax.set_box_aspect((np.ptp(xl),np.ptp(yl),np.ptp(zl)),zoom=1.12);ax.view_init(elev=28,azim=-58)
    ax.grid(False)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False;axis.pane.set_edgecolor('#FFFFFF')
    top.set(xlim=xl,ylim=yl,xlabel='X (m)',ylabel='Y (m)');top.set_aspect('equal',adjustable='box')
    fig.text(.25,.89,'(a) 三维执行轨迹',ha='center',fontsize=11)
    fig.text(.76,.89,'(b) 障碍位置与轨迹俯视图',ha='center',fontsize=11)
    top.grid(color='#E8EBEE',lw=.5,zorder=0)
    handles,labels=top.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,1.03))
    fig.subplots_adjust(left=.02,right=.99,bottom=.12,top=.84,wspace=.22)
    for ext in ['pdf','svg','png']:fig.savefig(out/('图9_Gazebo成功轨迹与真实障碍.'+ext),dpi=400,bbox_inches='tight')
    plt.close(fig)
    receipt=dict(selected=chosen,selection_rank=a.rank,all_reached_candidates=sum(set(v)=={'P0','P6','EGO'} for v in groups.values()),
        candidates_with_verified_geometry=len(candidates),geometry_sha256=sha(chosen['geometry']),obstacles=obstacles,sources=sources,
        selection_rule='All three monitor-reached in same task/repeat within paper40; verified identical geometry hash. Candidates ranked by worst method total turning angle per metre on 0.25m arc-length samples, then mean. Explicit rank chosen for obstacle-layout clarity as well as smoothness; not claimed as the unique smoothest or representative case.',
        display_rule='Unfiltered state positions after first logged navigation command through first 1.5m 3D goal entry; only linear boundary interpolation. No trajectory smoothing. Local odom positions translated to world by adding condition.start_pos.xy; z unchanged (SPAWN_Z=0). All scene obstacles at archived world coordinates.',
        interpretation='Selected successful illustrative trial for visual clarity; not representative of overall performance. Top view is obstacle ground projection; 1.5m circle is the XY projection of goal range.')
    (out/'图9选例与来源.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(chosen,ensure_ascii=False));print('candidates',len(candidates),'obstacles',len(obstacles))


if __name__=='__main__':main()
