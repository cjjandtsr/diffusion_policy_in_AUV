"""Build existing-data figures and a conservative revision of uploaded v4.2."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docx import Document

from analyze_paper_retention import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True)
    p.add_argument('--manuscript', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    z = zipfile.ZipFile(a.package)
    def raw(s):
        return z.read(next(i for i in z.infolist() if i.filename.endswith(s)))
    def csvrows(s):
        return list(csv.DictReader(io.StringIO(raw(s).decode('utf-8-sig'))))
    def dump(name, data):
        (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    def table(name, rows):
        with (out/name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig, name):
        fig.tight_layout()
        for ext in ['pdf','svg','png']:
            fig.savefig(out/(name+'.'+ext), dpi=300, bbox_inches='tight')
        plt.close(fig)
    rows = [r for r in csvrows('kinematic_episodes.csv') if r['计入正文统计']=='是']
    records = [dict(policy=r['策略编号'],task_id=r['任务ID'],world_id=r['场景ID'],
                    repeat=int(r['重复']),paper_safe_arrival=r['结局_1.5m']=='成功') for r in rows]
    ret = [summarize(records,'P2',x) for x in ['P6','P4','P3']]
    # Also compute the direct regularized-versus-unregularized comparison.
    dump('retention.json', ret+[summarize(records,'P4','P6')])
    labels = ['本文方法','无BC正则','单阶段DPPO']
    fig, ax = plt.subplots(figsize=(8.2,3.4))
    for i,r in enumerate(ret):
        c=r['counts']
        for value,left,color,label in [(-c['lost'],0,'#CA7979','丢失'),
              (c['retained'],0,'#537DB8','保持'),(c['new'],c['retained'],'#A7BFE4','新增')]:
            ax.barh(i,value,left=left,color=color,label=label if i==0 else None,edgecolor='white')
        ax.text(-c['lost']-2,i,str(c['lost']),ha='right',va='center')
        ax.text(c['retained']/2,i,str(c['retained']),ha='center',va='center',color='white')
        ax.text(c['retained']+c['new']+2,i,'+'+str(c['new']),va='center')
    ax.set_yticks(range(3),labels); ax.invert_yaxis(); ax.set_xlim(-45,150)
    ax.axvline(0,color='#555555',lw=1); ax.axvline(123,color='#777777',ls='--',lw=1)
    ax.set_xlabel('任务数（第一阶段已完成123个；共186任务）')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.24),ncols=3,frameon=False)
    save(fig,'图7_任务保持与丢失')
    pols=['P0','P2','P3','P4','P6']
    names=['UAV-iDP3','仅第一阶段','单阶段DPPO','无BC正则','本文方法']
    grouped=defaultdict(list)
    for r in rows: grouped[r['策略编号'],r['任务ID']].append(r)
    tasks=[]
    for (policy,task),rs in grouped.items():
        c=Counter(r['结局_1.5m'] for r in rs)
        outcome='成功' if c['成功']>=3 else ('碰撞' if c['碰撞']>=c['超时'] else '超时')
        tasks.append(dict(policy=policy,task_id=task,world_id=rs[0]['场景ID'],category=rs[0]['难度'],outcome=outcome))
    table('task_outcomes.csv',tasks)
    result=[]
    for policy,name in zip(pols,names):
        for cat in ['总体','稀疏','适中','密集']:
            subset=[r for r in tasks if r['policy']==policy and (cat=='总体' or r['category']==cat)]
            c=Counter(r['outcome'] for r in subset)
            result.append(dict(方法=name,难度=cat,任务=len(subset),安全到达=c['成功'],碰撞=c['碰撞'],超时=c['超时']))
    table('现有策略结局统计.csv',result)
    fig,axes=plt.subplots(1,3,figsize=(10.2,4.1))
    for ax,cat in zip(axes,['稀疏','适中','密集']):
        rr=[r for r in result if r['难度']==cat]; left=np.zeros(5)
        for field,color,hatch in [('安全到达','#537DB8',''),('碰撞','#CA7979','//'),('超时','#D9C68E','..')]:
            values=np.array([r[field]/r['任务']*100 for r in rr])
            ax.barh(range(5),values,left=left,color=color,hatch=hatch,label=field,edgecolor='white');left+=values
        ax.set_yticks(range(5),names if ax is axes[0] else ['']*5);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_title(cat+'（n='+str(rr[0]['任务'])+'）');ax.set_xlabel('任务占比（%）')
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,ncols=3,loc='upper center',bbox_to_anchor=(.5,1.08),frameon=False)
    save(fig,'现有策略_三类难度结局构成')
    blocks=csvrows('curve_blocks.csv')
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.6))
    for policy,name,col in [('P2_P3','单阶段DPPO','#7A5195'),('P4','无BC正则','#D99036'),('P6','本文方法','#537DB8')]:
        rr=[r for r in blocks if r['策略']==policy]
        if policy!='P2_P3': rr=[r for r in blocks if r['策略']=='P2_P3'][:8]+rr
        x=[int(r['轮'].split('-')[-1]) for r in rr]
        axes[0].plot(x,[float(r['采集到达率%']) for r in rr],label=name,color=col)
        axes[1].plot(x,[float(r['KL中位数']) for r in rr],label=name,color=col)
    for ax in axes: ax.axvline(128,color='#666666',ls='--');ax.set_xlabel('更新轮数');ax.legend()
    axes[0].set_ylabel('训练采集到达率（%）');axes[0].set_ylim(0,60)
    axes[1].set_ylabel('近似KL散度');axes[1].set_yscale('log')
    save(fig,'图8_训练过程')
    # Keep task-level Gazebo statistics exactly as supplied in the paper package.
    table('Gazebo40_正文结果.csv',csvrows('table5_task.csv'))
    # Preserve the uploaded DOCX's math, images and formatting; edit only w:t nodes.
    doc=Document(a.manuscript)
    changes=[]
    replacements=[
      ('使强化学习能够有效优化滚动执行的扩散策略','用于引导滚动执行的扩散策略绕行与提前避障'),
      ('不会把策略拉回示范的能力水平','意在避免把策略拉回示范的能力水平'),
      ('使强化学习能够引导策略绕行与提前避障','用于引导策略绕行与提前避障'),
      ('无人机只执行序列中的4步（称为执行前缀）后即重新规划','生成序列的前两步与历史观测帧对齐，不执行；无人机实际执行第3—6步（称为执行前缀）后即重新规划'),
      ('以示范预训练得到的UAV-iDP3为基础策略','EGO-Planner[7]同时是本文专家示范的来源。以示范预训练得到的UAV-iDP3为基础策略'),
      ('Single-stage DPPO不加正则，连续训练256轮','Single-stage DPPO使用本文回报与LoRA，是不分阶段、不加正则、连续训练256轮的变体'),
      ('并在软件在环仿真中得到验证','软件在环仿真中的变化方向一致，但幅度较小'),
      ('微调结果对训练随机性较为敏感，以不同随机种子重复训练时提升幅度较小，这也是深度强化学习中普遍存在的问题[38]；',''),
      ('后续工作将研究降低微调对随机性敏感程度的方法，引入更长的观测记忆与跟踪误差建模以减少碰撞','后续工作将引入更长的观测记忆与跟踪误差建模以减少碰撞'),
    ]
    for paragraph in doc.paragraphs:
        for old,new in replacements:
            nodes=paragraph._p.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
            text=''.join(n.text or '' for n in nodes)
            if old not in text: continue
            start=text.index(old);end=start+len(old);cursor=0;inserted=False
            for node in nodes:
                s=node.text or ''; lo,hi=cursor,cursor+len(s);cursor=hi
                if hi<=start or lo>=end: continue
                prefix=s[:max(0,start-lo)];suffix=s[max(0,end-lo):] if end<hi else ''
                node.text=prefix+(new if not inserted else '')+suffix;inserted=True
                node.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
            changes.append(dict(原句=old,改句=new,位置=paragraph.text[:80]))
    doc.save(out/'论文初稿_v4_3_阶段A修订.docx')
    table('正文修改对照.csv',changes)
    dump('source_identity.json',dict(package_sha256=hashlib.sha256(Path(a.package).read_bytes()).hexdigest(),
         manuscript_sha256=hashlib.sha256(Path(a.manuscript).read_bytes()).hexdigest(),
         scope='Existing 186-task paper labels; no new planner/timing experiments. DOCX edits preserve math/images; figure replacement and final renumbering pending integration.',
         changes=len(changes)))
    print(json.dumps(dict(output=str(out),changes=len(changes),tasks=len({r['task_id'] for r in tasks})),ensure_ascii=False))


if __name__=='__main__':
    main()
