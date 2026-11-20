"""Replace only Fig.9 and its caption in a new, preserved DOCX version."""
import argparse
import hashlib
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manuscript',required=True);p.add_argument('--figure-directory',required=True)
    p.add_argument('--output-name',default='论文初稿_v4_3_图9修订版.docx')
    a=p.parse_args();root=Path(a.figure_directory);doc=Document(a.manuscript)
    captions=[p for p in doc.paragraphs if p.text.startswith('图9')]
    assert len(captions)==1
    caption=captions[0];previous=caption._p.getprevious()
    assert previous.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing')
    previous.getparent().remove(previous)
    element=OxmlElement('w:p');caption._p.addprevious(element)
    new=Paragraph(element,caption._parent)
    new.add_run().add_picture(str(root/'图9_Gazebo成功轨迹与真实障碍.png'),width=Inches(6.1))
    caption.text='图9　同一Gazebo任务中三种方法共同成功的执行轨迹及真实障碍几何。(a)三维执行轨迹；(b)等比例俯视图，灰色区域为障碍在水平面的投影。方块为任务起点，星号为目标，彩色圆点为各方法首次进入1.5 m目标范围的位置，虚线圆为目标范围的水平投影。轨迹由原始位置记录转换到世界坐标，仅截取导航航段，未作平滑处理。该共同成功案例用于直观展示，不替代总体评价。'
    caption.add_run(' EGO位置日志从飞行途中开始，缺失的起始航段未补画。')
    target=root/a.output_name
    assert not target.exists()
    doc.save(target)
    original=Document(a.manuscript)
    tag='{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'
    assert len(list(original.element.iter(tag)))==len(list(doc.element.iter(tag)))==85
    (root/'稿件替换记录.json').write_text(json.dumps(dict(source=a.manuscript,
        source_sha256=hashlib.sha256(Path(a.manuscript).read_bytes()).hexdigest(),
        output=str(target),changed='Figure 9 image and caption only',math_objects_preserved=85),ensure_ascii=False,indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':main()
