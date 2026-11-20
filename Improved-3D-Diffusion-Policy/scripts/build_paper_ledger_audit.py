"""Read-only paper/workbook ledger, citation-order copy and source evidence export."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import openpyxl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def citations(text):
    return [int(n) for m in re.finditer(r'\[(\d+(?:\s*[,，]\s*\d+)*)\]',text)
            for n in re.split(r'\s*[,，]\s*',m.group(1))]


def renumber(main, supplement, output):
    body,bibliography=main.split('## 参考文献',1)
    references={int(m.group(1)):m.group(2).strip() for m in
                re.finditer(r'^\[(\d+)\]\s+(.+?)(?=\n\s*\n\[\d+\]|\Z)',bibliography,re.M|re.S)}
    # Working-draft notes mention future numbering, but are not first citations.
    scientific='\n'.join(line for line in body.splitlines() if not line.startswith('>'))
    order=list(dict.fromkeys(citations(scientific)))
    order+=list(dict.fromkeys(n for n in citations(supplement) if n not in order))
    uncited=sorted(set(references)-set(order));order+=uncited
    mapping={old:i+1 for i,old in enumerate(order)}
    def replace(text):
        return re.sub(r'\[(\d+(?:\s*[,，]\s*\d+)*)\]',lambda m:
            '['+','.join(str(mapping[int(x)]) for x in re.split(r'\s*[,，]\s*',m.group(1)))+']',text)
    updated=replace(body)+'## 参考文献\n\n'+'\n\n'.join('[%d] %s'%(mapping[n],references[n]) for n in order)+'\n'
    (output/'论文大稿_v3_参考文献顺序核对版_v1_0.md').write_text(updated,encoding='utf-8')
    (output/'补充材料_参考文献顺序核对版_v1_0.md').write_text(replace(supplement),encoding='utf-8')
    rows=[dict(原编号=n,新编号=mapping[n],状态='未在正文或补充材料引用，保留末尾' if n in uncited else '按正文首次引用，继以补充材料',文献=references[n]) for n in order]
    csv_write(output/'参考文献新旧编号对照.csv',rows)
    first=list(dict.fromkeys(citations('\n'.join(x for x in updated.split('## 参考文献')[0].splitlines() if not x.startswith('>')))))
    assert first==list(range(1,len(first)+1))
    return dict(references=len(references),uncited=uncited,BDPO_new=mapping[36],body_first_citations=first)


def ledger(snapshot, output, workbook):
    w=openpyxl.load_workbook(workbook,read_only=True,data_only=True)
    source=str(workbook);h=sha(workbook);rows=[]
    script='scripts/build_paper_ledger_audit.py'
    def add(location,label,value,unit,source_ref,source_hash=h,status='已对照数据表；未替代原始记录验收',raw=''):
        rows.append(dict(台账编号='N%04d'%(len(rows)+1),位置=location,指标=label,数值=value,单位=unit,
                         来源=source_ref,来源原值=raw,来源SHA256=source_hash,生成脚本=script,核对状态=status))
    locations={'表3 主结果':'正文表3、摘要、1引言、4.2、5结论；图5/图6',
        '表4 消融':'正文表4、4.2消融','主比较':'正文摘要、1引言、4.2；图5误差线不取此差值区间',
        '训练过程':'正文4.2训练过程、图7','直线参考':'正文4.1、图5直线参考',
        'S2 0.5m':'补充材料S2表S2','S3 第二次运行':'补充材料S3表S3、正文5局限'}
    for s in w:
        if s.title not in locations:continue
        headers={c.column:str(c.value or '') for c in next(s.iter_rows(min_row=2,max_row=2))}
        for row in s.iter_rows(min_row=3):
            context=' / '.join(str(c.value) for c in row[:2] if c.value is not None)
            for cell in row:
                if isinstance(cell.value,str) and cell.value.startswith('[') and cell.value.endswith(']'):
                    bounds=json.loads(cell.value)
                    for side,value in zip(('下界','上界'),bounds):
                        add(locations[s.title],context+' / '+headers[cell.column]+side,value,'原表单位',
                            source+'#'+s.title+'!'+cell.coordinate,raw=cell.value)
                    continue
                if type(cell.value) not in (int,float) and not (s.title=='主比较' and cell.column==5 and cell.value):continue
                unit='%' if headers[cell.column].endswith('率') else '百分点' if s.title=='主比较' and cell.column>=4 else '原单位'
                value=round(cell.value*100,1) if unit=='%' else cell.value
                add(locations[s.title],context+' / '+headers[cell.column],value,unit,
                    source+'#'+s.title+'!'+cell.coordinate,raw=cell.value)
    s=w['方法与设置']
    for i in range(3,23):
        value=s.cell(i,3).value
        add('正文表1/表2/3.6；补充S1.6',str(s.cell(i,2).value),value,'参数组',source+'#方法与设置!C'+str(i),status='数据表值；实现一致性见源码核对表')
    s=w['图9选例']
    for i in range(3,6):
        add('正文图8',str(s.cell(i,1).value)+'示例重复编号',s.cell(i,4).value,'次',source+'#图9选例!D'+str(i))
    for name,value in [('训练世界',21),('训练外世界',42),('任务',199),('稀疏任务',59),('适中任务',70),('密集任务',70),('每方法回合',995)]:
        ref=source+'#说明!A2' if name!='训练世界' else 'wsl_snapshot/routea_a2_selective_keep_run01/reference/tasks.json#distinct world_id'
        add('正文摘要/3.6/4.1、表3表注',name,value,'个',ref,source_hash=h if name!='训练世界' else sha(Path.cwd()/'wsl_snapshot/routea_a2_selective_keep_run01/reference/tasks.json'))
    add('正文4.2消融','第二阶段相对第一阶段成功率差',7.5,'百分点',source+'#表3 主结果!G11-G7',raw='100*(648-573)/995=7.537688442211055')
    add('补充S3','第二次运行第二阶段相对第一阶段成功率差',18.2,'百分点',source+'#S3 第二次运行!G7-G3',raw='100*(398-217)/995=18.190954773869347')
    for name in ['论文大稿_v3.md','补充材料.md']:
        f=snapshot/name;text=f.read_text('utf-8')
        for line_no,line in enumerate(text.splitlines(),1):
            for item in re.findall(r'【[^】]*(?:待|GPU|Gazebo)[^】]*】',line):
                add(name+':'+str(line_no),item,'待补','',str(f)+':'+str(line_no),sha(f),'待补，未估填')
        # Empty path-efficiency cells inherit the same table-wide missing-data status.
        active=None;last_class=''
        for line_no,line in enumerate(text.splitlines(),1):
            if '**表3' in line:active='表3'
            if '**表4' in line:active='表4'
            if active and line.startswith('|') and not line.startswith('| ---'):
                cells=[x.strip() for x in line.split('|')[1:-1]]
                if cells and cells[-1]=='' and len(cells)>=5:
                    add(name+':'+active+'行'+str(line_no),'路径效率 '+(' / '.join(cells[:2])),'待补','',str(f)+':'+str(line_no),sha(f),'待成功轨迹1m重算')
            if active and (line.startswith('![') or line.startswith('表注')):active=None
    add('补充图S1','第二次训练曲线来源','待数据表收录','',
                str(snapshot/'研究报告/analysis_20261108/oprime/curve_blocks.csv'),'','数据表缺此sheet；独立CSV仅作候选来源')
    ci_path=snapshot/'研究报告/analysis_20261108/arrival_1m/arrival_1m_main.csv'
    names={'P0':'UAV-iDP3','P2':'Reg-DPPO w/o Stage II','P6':'UAV-iDP3 + Reg-DPPO'}
    with ci_path.open(encoding='utf-8-sig') as f:
        for i,row in enumerate(csv.DictReader(f),2):
            if row['策略'] not in names:continue
            for side,value in zip(('下界','上界'),json.loads(row['区间'])):
                add('正文图5',names[row['策略']]+' / '+row['难度']+' 世界聚类95%区间'+side,
                    value,'%',str(ci_path)+':'+str(i)+'#区间',sha(ci_path),'按指定1m CSV读取')
    commit=(snapshot.parent/'paper_commit.txt').read_text('utf-8').strip()
    for row in rows:row['论文commit']=commit
    csv_write(output/'数字台账.csv',rows)
    return len(rows)


def source_audit(workspace, output):
    native=workspace/'wsl_snapshot/routea_test12_independent_run02/results/native_a2'
    base='diffusion_policy_3d/rl/dppo_uav_rebuild/'
    specs=[]
    def add(item,claim,path,start,end,status,note):
        f=native/path;lines=f.read_text('utf-8').splitlines()
        specs.append(dict(项目=item,论文表述=claim,判定=status,说明=note,文件=str(f),sha256=sha(f),行号=f'{start}-{end}',片段='\n'.join('%d %s'%(i,lines[i-1]) for i in range(start,min(end,len(lines))+1))))
    add('式1','9D状态与3帧', 'diffusion_policy_3d/env/kinematic_uav_env.py',913,924,'一致','环境缓存8帧；策略取最新3帧，不能把缓存首维8当并行环境数。')
    add('式2/式3','x0扩散加噪/预测',base+'engine/selective_keep.py',107,122,'一致','此处验证后训练正则中的加噪；预训练MSE另见phase1a_policy。')
    add('式2/式3预训练','DDPM加噪与sample预测MSE','diffusion_policy_3d/policy/phase1a_policy.py',802,820,'一致','add_noise后sample分支target=trajectory，逐元素MSE后取均值；BC的实际训练完成轮数另缺原始训练日志。')
    add('式4/S1/S2','归一化softmin与softplus几何损失','diffusion_policy_3d/model/common/collision_loss.py',145,167,'一致','有sqrt(d²+eps)数值floor；归一化log(k)/tau修正与非负clamp存在。')
    add('式4/S2','几何损失上限与时序权重','diffusion_policy_3d/model/common/collision_loss.py',256,285,'一致','margin先限幅，softplus后再限幅2*d_safe，指数权重归一化。')
    add('式5/表1终止进展时间','+100/-50/10预算/-0.05', 'diffusion_policy_3d/env/reward_v2_support/reward.py',118,155,'一致','进展沿实际连续执行路径；失败还覆盖无效/越界等硬异常，正文只列正常范围的碰撞/触地。')
    add('式6/表1风险','prefix/tail障碍与触地barrier','diffusion_policy_3d/env/reward_v2_support/reward.py',70,90,'一致','0.45/0.10、1.0/0.20、前4段1、后4段指数衰减，取最大风险。')
    add('式5/S1.2/表1绕行','梯形积分、名义72s、预算1', 'diffusion_policy_3d/env/reward_v2_support/reward.py',13,29,'一致','函数返回负cost，正文将C定义正量并减去，符号等价。')
    add('式7','LoRA W0+alpha/r BA',base+'actor_ddpm/lora.py',35,101,'一致','Linear与Conv1d的LoRA实现，原参数冻结、B零初始化；后训练层数见已存network_spec实测。')
    add('式8/S3/S4','GAE/bootstrap/return',base+'engine/gae.py',6,31,'正文需补说明，补充一致','正文单mask写法未表达批次中途结束m^b=1而m^c=0；S3已区分。硬终止和期限不自举。')
    add('式9','H8×11对数概率平均',base+'engine/official_loss_components.py',76,95,'不完整','实际先将每元素logprob clamp到[-5,2]，再取[2:10]的8×11均值。正文式9遗漏这一截断。')
    add('式9配置','reward_start=2/reward_horizon=8',base+'entrypoints/build_system.py',109,123,'一致','v2明确H8，非只用执行前缀H4。')
    add('式10/S1.4','PPO最小目标与指数clip调度',base+'engine/official_loss_components.py',83,125,'一致','优势按外层批统计标准化，去噪折扣0.8，clip基值0.001至0.1。')
    add('式11/S5','BC正则与1/16尺度',base+'engine/selective_keep.py',107,122,'正文不完整，补充一致','实际loss先对16×11加权误差求和再除16。正文式11无1/16，S5包含；应由Claude统一定义，λ沿用实测不能静默改尺度。')
    add('式11/S5','执行前缀位置yaw掩码',base+'engine/selective_keep.py',11,28,'一致','完整段仅[2:6]×通道[0,1,2,9]；中途到达的部分段只位置、权重fraction²。')
    add('S1.5','任务/来源/状态均匀抽样与batch32',base+'engine/selective_keep.py',95,105,'一致','先均匀task，再来源和样本；不是三类别各1/3。')
    add('S1.5/λ','四批标定0.1中位数比',base+'engine/selective_keep.py',148,179,'一致','λ=0.1*median(PPO norm)/median(keep norm)，非每批比值都为0.1；P6原标定0.01061006086489362。')
    vendor='third_party/dppo_official_cc7234a/'
    add('S6','sample→epsilon代数换算',vendor+'model/diffusion/diffusion_vpg.py',192,221,'一致','actual direct-x0分支在clamp前由x0计算epsilon方向项；E5ActorBackend不先转换以避免往返漂移。clip后不重新计算epsilon。')
    add('S6/表2','likelihood std floor0.1',vendor+'model/diffusion/diffusion_vpg.py',353,399,'一致','get_logprobs使用min_logprob_denoising_std；训练采样另有min_sampling下限。')
    add('S6/表2','deterministic与采样std floor',vendor+'model/diffusion/diffusion_vpg.py',310,337,'需区分语义','deterministic DDIM采样std=0；0.1用于随机训练采样及logprob，不应说确定性评价仍添加0.1噪声。')
    add('S6/表2','训练clip±1，评价±2',vendor+'uav_idp3/e5_runtime.py',238,265,'训练一致','训练继承BC clip_sample_range默认1.0；评价2.0为外部受控override，见run_routea_xk_evaluation与E2包。')
    add('S1.3/价值网络','6维上下文',base+'engine/critic_state.py',23,42,'一致','beta、D0/10、起点偏移body/10、剩余比例共6维，不进入actor输入。')
    add('S1.3/表2','LayerNorm/位置嵌入/256×256 SiLU',base+'engine/value.py',13,59,'一致','episode_context映射到首隐藏层且零初始化。')
    csv_write(output/'中文稿实现逐项核对.csv',specs)
    report=['# 第3章与表1/表2、补充S1—S6源码核对','',
            '日期2026-11-08；论文来源截至de74d38，项目参照15ad24e2；权威计算源采用原P6 native_a2，不用当前F5/F6活动树。',
            '本报告不改论文论证或代码。关键差异由Claude改正文；其余参数组的配置对照见参数CSV。','']
    for row in specs:
        report += ['## '+row['项目']+'：'+row['判定'],row['说明'],
                   '`'+row['文件']+':'+row['行号']+'`；SHA256 `'+row['sha256']+'`','```python',row['片段'],'```','']
    report+=['## 1m结局语义',
             '需求是“碰撞前首次进入目标1m”。正文“飞行中未发生碰撞且进入目标1m”容易被读成后续碰撞会否决此前成功。按首次事件顺序，先进入1m者已经成功，不能被旧0.5m轨迹后面的碰撞推翻；此处是语义澄清建议，未直接改正文。原训练/回报仍在0.5m终止，不把主终点改写成训练半径。','',
             '## 状态变更','第3章核心公式核对完成；式9、式11及正文GAE需Claude确认修订。S6代数/实际direct-x0分支一致，std floor与确定性采样应分开叙述。未修改计算源码、原始轨迹或论文论断。']
    (output/'中文稿实现核对报告.md').write_text('\n'.join(report),encoding='utf-8')
    return len(specs)


def settings_audit(workspace,snapshot,output):
    w=openpyxl.load_workbook(snapshot/'研究报告/2026-11-08_论文数据表.xlsx',data_only=True)
    native=workspace/'wsl_snapshot/routea_test12_independent_run02/results/native_a2'
    network=workspace/'论文/图表/data/network_spec.json'
    config=workspace/'wsl_snapshot/routea_a2_selective_keep_run01/arms/T.yaml'
    # Every workbook parameter group is accounted for, including unverified inputs.
    refs={3:(None,'','未找到示范数据原件','988/60151/9:1仅为作者数据表值，不能据配置证明实际数据规模。'),
        4:(network,'21-35','一致','3帧/4096点/9D/16×11，按注册BC实测结构。'),
        5:(None,'','部分核对','运动学传感器沿用原评价；当前Gazebo实际相机参数未到，不把默认值当实测。'),
        6:(network,'21-47','一致','几何128、状态64、3帧拼接576。'),
        7:(network,'48-75','一致','U-Net 256/512/1024，核5，时间嵌入128。'),
        8:(network,'94-105','一致','注册BC的训练扩散50步与DDIM10步。'),
        9:(None,'','未找到实际BC完整cfg','目录名lambda0p5和当前候选配置不独立证明注册BC的辅助损失系数。'),
        10:(config,'environment; run; dppo','一致','0.3s/H8/H4/60，H8为概率比窗口、H4为实际执行，两者不得混同。'),
        11:(config,'reward; environment','一致','训练到达0.5m、碰撞代理0.1m；论文评价1m为离线终点。'),
        12:(network,'118-129','一致','rank8/alpha8，1.4296%以基础策略参数为分母。'),
        13:(native/'diffusion_policy_3d/rl/dppo_uav_rebuild/engine/value.py','13-59','一致','两个256隐藏层及SiLU。'),
        14:(config,'optimizer','一致','actor全局256轮线性2e-5至1e-5，critic1e-3。'),
        15:(config,'dppo','一致','gamma环境0.99、GAE0.95、去噪折扣0.8。'),
        16:(native/'diffusion_policy_3d/rl/dppo_uav_rebuild/engine/official_loss_components.py','94-104','一致','epsilon基值0.001、最大0.1，指数调度。'),
        17:(config,'run; dppo','一致','1024转移/5遍/20提交；微批128不是并行环境。'),
        18:(config,'run; a2.parent_checkpoint','一致','P2 u128父状态，T臂到u256，共两个128阶段。'),
        19:(workspace/'wsl_snapshot/routea_a2_selective_keep_run01/reference/inventory.json','available_tasks; selected_prefixes','一致','22+2+23=47任务，549前缀样本；draw_indices每批32见源码核对表。'),
        20:(workspace/'wsl_snapshot/routea_a2_selective_keep_run01/arms/T/keep_calibration.json','coefficient; batches','一致','0.01061006086489362，四批梯度中位数比标定目标0.1。'),
        21:(native/'diffusion_policy_3d/env/reward_v2_support/reward.py','118-155','一致','+100/-50/-0.05；实际终止细节见逐式报告。'),
        22:(workspace/'program/paper_data_request_20261108/paper_snapshot/研究报告/analysis_20261108/arrival_1m/arrival_1m_main.csv','1m到达; 区间','部分核对','1m来自作者离线重分析；确定性/clip2由既有E2工程包核验，非训练奖励半径改变。')}
    rows=[]
    for i in range(3,23):
        f,location,status,note=refs[i]
        rows.append(dict(位置='方法与设置!C'+str(i),参数=w['方法与设置'].cell(i,2).value,
            数据表值=w['方法与设置'].cell(i,3).value,判定=status,说明=note,
            来源=str(f) if f else '未找到对应运行直接原件',行号或字段=location,sha256=sha(f) if f else ''))
    csv_write(output/'表1表2参数组核对.csv',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',default=str(Path(__file__).resolve().parents[3]))
    a=p.parse_args();workspace=Path(a.workspace)
    snapshot=workspace/'program/paper_data_request_20261108/paper_snapshot'
    output=workspace/'program/paper_data_request_20261108/ledger_audit';output.mkdir(parents=True,exist_ok=True)
    workbook=snapshot/'研究报告/2026-11-08_论文数据表.xlsx'
    count=ledger(snapshot,output,workbook)
    refs=renumber((snapshot/'论文大稿_v3.md').read_text('utf-8'),(snapshot/'补充材料.md').read_text('utf-8'),output)
    audits=source_audit(workspace,output)
    settings_audit(workspace,snapshot,output)
    (output/'build_summary.json').write_text(json.dumps(dict(ledger_rows=count,citation_order=refs,audit_items=audits,source_workbook_sha256=sha(workbook)),indent=2,ensure_ascii=False),encoding='utf-8')
    print('PAPER_LEDGER_AUDIT_BUILT',count,audits,refs)


if __name__=='__main__':
    main()
