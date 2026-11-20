"""Integrate existing-evidence figures and approved edits without inventing results."""
import argparse
import csv
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args()
    root=Path(a.directory);doc=Document(root/'论文初稿_v4_3_阶段A修订.docx');log=[]
    def replace(par,old,new):
        nodes=par._p.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
        text=''.join(n.text or '' for n in nodes)
        if old not in text:return
        start=text.index(old);end=start+len(old);cursor=0;inserted=False
        for n in nodes:
            s=n.text or '';lo,hi=cursor,cursor+len(s);cursor=hi
            if hi<=start or lo>=end:continue
            n.text=s[:max(0,start-lo)]+(new if not inserted else '')+(s[end-lo:] if end<hi else '');inserted=True
        log.append(dict(old=old,new=new))
    def before(par,text='',image=None):
        element=OxmlElement('w:p');par._p.addprevious(element);new=Paragraph(element,par._parent)
        if image:new.add_run().add_picture(str(root/image),width=Inches(6.1))
        else:new.add_run(text)
        return new
    for par in list(doc.paragraphs):
        text=par.text
        if text.startswith('图4') and '两层决策' in text:
            previous=par._p.getprevious()
            if previous is not None and previous.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'):
                previous.getparent().remove(previous)
            par._p.getparent().remove(par._p);log.append(dict(old='原图4及图注',new='删除；执行索引已补入3.1节'));continue
        replace(par,'对应图4序列的第m+2步','对应生成序列的第m+2步')
        replace(par,'（图4）','')
        replace(par,'图5','图4')
        replace(par,'图7','图8')
        replace(par,'提交规划序列前4个路径点构成的轨迹','提交规划窗口执行前缀的4个路径点构成的轨迹')
        replace(par,'在接近障碍之前即调整航向，与障碍保持距离并飞向目标','在接近障碍时调整航向并飞向目标')
        replace(par,'这说明微调使策略在示范未覆盖的状态下学会了绕行并到达目标，而不是以更冒险或更绕远的飞行换取成功。','本次训练改善了固定任务集上的目标到达表现，其中既包含稀疏任务中的到达改善，也包含受阻任务中的绕行改善；总体碰撞率与路径效率没有显示相应代价增加。')
        if par.text.startswith('图2'):
            previous=par._p.getprevious()
            fills=previous.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/picture}blipFill') if previous is not None else []
            for fill in fills:
                crop=OxmlElement('a:srcRect');crop.set('t','33000');fill.insert(1,crop)
            if fills:
                for extent in previous.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent')+previous.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm/{http://schemas.openxmlformats.org/drawingml/2006/main}ext'):
                    extent.set('cy',str(round(int(extent.get('cy'))*.67)))
                par.add_run(' 本图保留编码器、U-Net和LoRA子图；整体网络流程见图1。')
        if par.text.startswith('图6') and '俯视' in par.text:
            previous=par._p.getprevious()
            if previous is not None and previous.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'):
                previous.getparent().remove(previous)
            before(par,image='图6_真实三维轨迹与目标接近.png')
            par.text='图6　固定任务中的真实三维执行轨迹与目标接近过程。灰色虚线为UAV-iDP3，蓝色实线为本文方法，灰色半透明体为障碍，方块为起点，星号为目标，圆点为到达终点，叉号为失败终点；下排为目标距离随累计执行路长的变化，虚线表示1.5 m范围。'
        if par.text.startswith('图8给出'):
            before(par,image='图7_任务保持与丢失.png')
            before(par,'图7　以第一阶段策略为参照的任务保持、丢失与新增完成。参照已完成123个任务；完整评价集共186任务。虚线标出123个参照成功任务。')
            par.text='图8给出微调过程中的训练采集成功率与近似KL散度。第128轮之后，完整方法保持了较平稳的采集表现；不加正则继续训练出现较大的KL波动，单阶段训练后段的采集成功率下降。这些训练过程记录与本次任务保持分析相互补充，训练采集成功率不直接作为确定性评价结果。'
        if par.text.startswith('图8') and '每16轮' in par.text:
            previous=par._p.getprevious()
            if previous is not None and previous.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'):
                previous.getparent().remove(previous)
            before(par,image='图8_训练过程.png')
        if par.text.startswith('4.5 讨论'):
            before(par,image='图9_Gazebo实际轨迹与位置差.png')
            before(par,'图9　同一固定Gazebo任务的一次实际执行记录。左为三种方法的实际轨迹，右为记录时间重叠范围内参考位置与实际位置之差；参考位置线性插值到状态记录时刻。该案例不替代总体评价。')
        if par.text.startswith('EGO-Planner[7]同时是'):
            par.add_run(' 补充运动学对照采用EGO-Planner、Fast-Planner的kino模式和基于3DVFH+的PX4 local_planner公开实现[39]。三个规划器使用相同传感器配置产生的局部观测、自身位姿与目标，不读取障碍真值或全局地图；环境采用理想跟踪，计算期间仿真不推进，耗时单独测量。EGO采用项目保存的采集入口配置作为参数基准，Fast与PX4原生速度和加速度参数分别设为0.5 m/s和0.25 m/s²，其余必要配置只在训练或验证场景确定，测试前冻结。PX4采用原生航点生成与理想跟踪时间转换，其加速度参数用于原生限速计算，不代表接口已施加与Fast样条相同的轨迹加速度约束。')
    doc.add_paragraph('[39] PX4. PX4-Avoidance: local_planner[CP/OL]. Git commit c386852c1b8e2084094e194d9dcd46fdd0c01646 [2026-11-10]. https://github.com/PX4/PX4-Avoidance. 基于3DVFH+的公开实现；算法背景见[24]。')
    doc.add_paragraph('附图S1　本文方法在固定评价任务中的真实失败案例。叉号表示碰撞终止位置；所示记录来自图6同一登记运行目录，任务身份与原轨迹路径见failure_case.json。')
    doc.add_picture(str(root/'附图_本文方法真实失败案例.png'),width=Inches(6.1))
    # Add CI to the existing retention paragraph; computed from this frozen task set.
    ci=json.loads((root/'retention.json').read_text(encoding='utf-8'))
    for par in doc.paragraphs:
        if '只进行第一阶段微调时任务成功率为66.1%' in par.text:
            lo,hi=npct(ci[0]['paired_success_rate_difference_ci95'])
            par.add_run(f' 在这186个任务中，完整方法相对第一阶段净增12个成功任务，配对成功率差为6.5个百分点，按40个场景聚类重抽样20000次的95%区间为[{lo:.1f}, {hi:.1f}]个百分点。')
    doc.save(root/'论文初稿_v4_3_桌面修订版.docx')
    (root/'整合修改记录.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Integrated',len(log),'edits and real-data figures')


def npct(values):return [100*x for x in values]


if __name__=='__main__':main()
