"""Paper figures 5--8 and S1 from the dated author workbook and raw trajectories.

Read-only inputs; a separate output directory preserves all earlier figures.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import openpyxl

METHODS = ['UAV-iDP3', 'Reg-DPPO w/o Stage II', 'UAV-iDP3 + Reg-DPPO']
COLORS = ['#7C8796', '#E69F00', '#0072B2']
CATEGORIES = ['总体', '稀疏', '适中', '密集']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, output, name):
    fig.savefig(output/(name+'.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output/(name+'.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def style():
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
        'font.size':9, 'axes.labelsize':9, 'legend.fontsize':8, 'xtick.labelsize':8,
        'ytick.labelsize':8, 'pdf.fonttype':42, 'ps.fonttype':42,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7})


def success(workbook, ci_path, output):
    rows = list(workbook['表3 主结果'].values)[2:]
    data = {(r[0],r[1]):r for r in rows}
    with ci_path.open(encoding='utf-8-sig') as f:
        ci = {(r['策略'],r['难度']):json.loads(r['区间']) for r in csv.DictReader(f)}
    line = {r[0]:r[3]*100 for r in list(workbook['直线参考'].values)[2:]}
    fig,ax = plt.subplots(figsize=(7.8,4.1))
    x = np.arange(4); width=.23
    for i,(key,policy,label,color) in enumerate(zip(
            ['UAV-iDP3','Reg-DPPO w/o Stage II','本文方法'], ['P0','P2','P6'], METHODS,COLORS)):
        values = np.array([data[key,c][3]/data[key,c][2]*100 for c in CATEGORIES])
        limits = np.array([ci[policy,c] for c in CATEGORIES])
        ax.bar(x+(i-1)*width,values,width,color=color,label=label,
               edgecolor='white',linewidth=.5,zorder=3)
        ax.errorbar(x+(i-1)*width,values,yerr=np.array([values-limits[:,0],limits[:,1]-values]),
                    fmt='none',ecolor='#243447',capsize=3,elinewidth=.8,zorder=4)
    for i,c in enumerate(CATEGORIES):
        ax.hlines(line[c],i-.39,i+.39,color='#333333',linestyle='--',linewidth=1.1,
                  label='直线飞行参考（按任务）' if i==0 else None,zorder=5)
    ax.set_xticks(x,CATEGORIES);ax.set_ylim(0,104);ax.set_yticks(np.arange(0,101,20))
    ax.set_ylabel('成功率（%）');ax.grid(axis='y',color='#E2E5E8',linewidth=.6,zorder=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False)
    fig.tight_layout();save(fig,output,'figure_05_success_by_class')


def outcomes(workbook,output):
    data={(r[0],r[1]):r for r in list(workbook['表3 主结果'].values)[2:]}
    fig,axes=plt.subplots(1,3,figsize=(8.2,3.9),sharey=True)
    labels=['UAV-iDP3','Reg-DPPO\nw/o Stage II','UAV-iDP3\n+ Reg-DPPO']
    for ax,c in zip(axes,CATEGORIES[1:]):
        bottom=np.zeros(3)
        for field,label,color,hatch in [(3,'成功','#56B4A1',''),(4,'碰撞','#D88A4A','//'),(5,'超时','#B9BEC7','..')]:
            value=np.array([data[m,c][field]/data[m,c][2]*100 for m in ['UAV-iDP3','Reg-DPPO w/o Stage II','本文方法']])
            ax.bar(np.arange(3),value,.65,bottom=bottom,color=color,edgecolor='white',linewidth=.5,
                   label=label,hatch=hatch)
            bottom+=value
        ax.set_title(c,fontsize=10);ax.set_xticks(np.arange(3),labels);ax.set_ylim(0,100)
        ax.grid(axis='y',color='#E2E5E8',linewidth=.6);ax.set_axisbelow(True)
    axes[0].set_ylabel('回合比例（%）')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,output,'figure_06_outcomes')


def training(workbook,blocks_path,output):
    rows=list(workbook['训练过程'].values)[2:]
    series={}
    for label,window,rate,kl in rows:
        series.setdefault(label,[]).append((int(window.split('-')[1]),rate*100,kl))
    first=series['第一阶段 + 不中断续训（同一运行）'][:8]
    branches=[('本文方法第二阶段',METHODS[2],COLORS[2],'o'),
              ('分叉不加正则（第二阶段）','Reg-DPPO w/o BC reg.','#E69F00','s'),
              ('第一阶段 + 不中断续训（同一运行）','Single-stage DPPO','#7A5195','^')]
    draw_training(first,[(series[k][-8:],name,color,marker) for k,name,color,marker in branches],
                  output,'figure_07_training')
    with blocks_path.open(encoding='utf-8-sig') as f: raw=list(csv.DictReader(f))
    def block(policy):
        return [(int(r['轮'].split('-')[1]),float(r['采集到达率%']),float(r['KL中位数']))
                for r in raw if r['策略']==policy]
    draw_training(block('P14'),[(block('P15'),METHODS[2]+'（第二次运行）',COLORS[2],'o')],
                  output,'figure_S1_second_run_training')


def draw_training(first,branches,output,name):
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.6))
    for i,ax in enumerate(axes):
        ax.plot([x[0] for x in first],[x[i+1] for x in first],color='#7C8796',marker='D',
                markersize=3,linewidth=1.4,label='第一阶段')
        for points,label,color,marker in branches:
            points=[first[-1]]+points
            ax.plot([x[0] for x in points],[x[i+1] for x in points],color=color,marker=marker,
                    markersize=3,linewidth=1.4,label=label)
        ax.axvline(128,color='#66727C',linestyle='--',linewidth=.8)
        ax.set_xlim(8,264);ax.set_xticks([16,64,128,192,256]);ax.set_xlabel('训练轮次（每16轮聚合）')
        ax.grid(color='#E2E5E8',linewidth=.6);ax.set_axisbelow(True)
    axes[0].set_ylabel('采集成功率（%）');axes[0].set_ylim(0,55);axes[0].set_title('(a)',loc='left',fontsize=10)
    axes[1].set_ylabel('近似KL（中位数）');axes[1].set_yscale('log');axes[1].set_title('(b)',loc='left',fontsize=10)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=2,frameon=False)
    fig.tight_layout();save(fig,output,name)


def first_entry(path,goal,radius=1.):
    goal=np.array(goal);output=[path[0]]
    if np.linalg.norm(path[0]-goal)<radius:return np.array(output)
    for a,b in zip(path,path[1:]):
        d=b-a;v=a-goal;aa=d@d;bb=2*(v@d);cc=v@v-radius**2
        disc=bb*bb-4*aa*cc
        if aa>0 and disc>=0:
            roots=[t for t in [(-bb-math.sqrt(disc))/(2*aa),(-bb+math.sqrt(disc))/(2*aa)] if 0<=t<=1]
            if roots:
                output.append(a+min(roots)*d);return np.array(output)
        output.append(b)
    return np.array(output)


def trajectories(data_path,output):
    raw=json.loads(data_path.read_text(encoding='utf-8'))
    fig,axes=plt.subplots(3,2,figsize=(8,8.1))
    for i,category in enumerate(CATEGORIES[1:]):
        records=[r for r in raw if r['selection']['category']==category]
        goal=np.array(records[0]['header']['task']['goal_pos'])
        paths=[]
        for r in records:
            path=np.array([pt for row in r['transitions'] for pt in row['executed_path']])
            paths.append((r['policy'],first_entry(path,goal)))
        for j,dimensions in enumerate([(0,1),(0,2)]):
            ax=axes[i,j]
            for obs in records[0]['header']['geometry']:
                if j==0:
                    if obs['shape'] in ('cylinder','sphere'):
                        patch=Circle((obs['x'],obs['y']),obs['radius'])
                    else:
                        from matplotlib.transforms import Affine2D
                        patch=Rectangle((-obs['size_x']/2,-obs['size_y']/2),obs['size_x'],obs['size_y'])
                        patch.set_transform(Affine2D().rotate(obs.get('yaw',0)).translate(obs['x'],obs['y'])+ax.transData)
                else:
                    if obs['shape']=='sphere':patch=Circle((obs['x'],obs['z']),obs['radius'])
                    else:
                        width=2*obs['radius'] if obs['shape']=='cylinder' else abs(math.cos(obs['yaw']))*obs['size_x']+abs(math.sin(obs['yaw']))*obs['size_y']
                        height=obs['height'] if obs['shape']=='cylinder' else obs['size_z']
                        patch=Rectangle((obs['x']-width/2,obs['z']-height/2),width,height)
                patch.set_facecolor('#9AA5B1');patch.set_edgecolor('#788391');patch.set_alpha(.24);patch.set_linewidth(.5);ax.add_patch(patch)
            for policy,path in paths:
                color=COLORS[0] if policy=='P0' else COLORS[2]
                ax.plot(path[:,dimensions[0]],path[:,dimensions[1]],color=color,
                        linestyle='--' if policy=='P0' else '-',linewidth=1.6,
                        label=METHODS[0] if policy=='P0' else METHODS[2])
                ax.scatter(*path[-1,list(dimensions)],s=20,color=color,marker='x' if policy=='P0' else 'o',zorder=4)
            start=paths[0][1][0]
            ax.scatter(*start[list(dimensions)],color='#212D38',marker='s',s=20,zorder=5,label='起点')
            ax.add_patch(Circle(goal[list(dimensions)],1.,fill=False,color='#2D8A65',linestyle=':',linewidth=1.1))
            ax.scatter(*goal[list(dimensions)],marker='*',color='#2D8A65',s=70,zorder=5,label='目标（1 m）')
            points=np.vstack([x[1] for x in paths]+[goal[None,:]])
            xmin,xmax=points[:,dimensions[0]].min()-2,points[:,dimensions[0]].max()+2
            ymin,ymax=points[:,dimensions[1]].min()-2,points[:,dimensions[1]].max()+2
            ax.set_xlim(xmin,xmax);ax.set_ylim((max(0,ymin),max(ymax,5.2)) if j==1 else (ymin,ymax))
            ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x（m）');ax.set_ylabel(('y' if j==0 else 'z')+'（m）')
            ax.set_title(category+' · '+('俯视' if j==0 else '侧视投影'),fontsize=10)
            ax.grid(color='#E7E9EC',linewidth=.4)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1),ncol=4,frameon=False)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,output,'figure_08_trajectories')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook',required=True);p.add_argument('--ci',required=True)
    p.add_argument('--curve-blocks',required=True);p.add_argument('--trajectories',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();style();output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    workbook=openpyxl.load_workbook(a.workbook,data_only=True)
    success(workbook,Path(a.ci),output);outcomes(workbook,output)
    training(workbook,Path(a.curve_blocks),output);trajectories(Path(a.trajectories),output)
    sources={name:dict(path=str(Path(value).resolve()),sha256=sha(value)) for name,value in vars(a).items() if name!='output'}
    (output/'figure_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PAPER_FIGURES_RENDERED',5)


if __name__=='__main__':main()
