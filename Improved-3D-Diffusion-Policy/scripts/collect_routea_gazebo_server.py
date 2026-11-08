"""Read-only server A0 inventory; does not launch, build, install or kill ROS jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','catkin','px4','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    roots={name:Path(getattr(a,name)).resolve() for name in ('repo','catkin','px4')}
    assert all(x.is_dir() for x in roots.values())
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    commands=[['uname','-a'],['nvidia-smi'],['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
              ['rosversion','-d'],['rosversion','gazebo_ros'],['gazebo','--version'],['rosnode','list'],['rostopic','list']]
    results=[]
    for i,argv in enumerate(commands):
        try:
            result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=12)
            row=dict(argv=argv,returncode=result.returncode,output=result.stdout)
        except (FileNotFoundError,subprocess.TimeoutExpired) as exc:
            row=dict(argv=argv,error=str(exc))
        results.append(row)
    files=[]
    candidates=list((roots['catkin']/'scripts').glob('*.sh'))
    candidates+=list((roots['catkin']/'scripts').glob('*monitor*.py'))
    candidates+=list((roots['catkin']/'src').rglob('*ml_planner*.py'))
    candidates+=list((roots['catkin']/'src').rglob('*ml_planner*.launch'))
    candidates+=list((roots['catkin']/'src').rglob('*routea*.py'))
    candidates+=list((roots['catkin']/'src').rglob('*routea*.launch'))
    candidates+=list((roots['catkin']/'src').rglob('package.xml'))
    for folder in (roots['px4']/'Tools/sitl_gazebo/models/iris_D435i',
                   roots['px4']/'Tools/sitl_gazebo/models/realsense_camera'):
        if folder.is_dir():
            candidates+=list(folder.rglob('*.sdf'))+list(folder.rglob('*.jinja'))
    for source in sorted(set(candidates)):
        if not source.is_file():continue
        group='catkin' if source.parts[:len(roots['catkin'].parts)] == roots['catkin'].parts else 'px4'
        relative=source.relative_to(roots[group]);target=out/'source'/group/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        files.append(dict(root=group,path=str(relative),sha256=sha(source),bytes=source.stat().st_size))
    worlds=roots['px4']/'Tools/sitl_gazebo/worlds/test'
    world_files=[dict(path=str(x.relative_to(roots['px4'])),sha256=sha(x))
                 for x in sorted(worlds.rglob('*')) if x.is_file() and x.suffix in ('.world','.json')] if worlds.exists() else []
    ckpt=roots['repo']/'data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01/T/checkpoint_update_000256'
    weights={name:dict(exists=(ckpt/name).is_file(),sha256=sha(ckpt/name) if (ckpt/name).is_file() else None)
             for name in ('state.pt','header.json','SHA256.json','COMMITTED.json')}
    for name in ('header.json','SHA256.json','COMMITTED.json'):
        if (ckpt/name).is_file():
            (out/'P6_metadata').mkdir(exist_ok=True)
            shutil.copy2(ckpt/name,out/'P6_metadata'/name)
    report=dict(roots={k:str(v) for k,v in roots.items()},commands=results,source=files,test_worlds=world_files,
                P6_checkpoint=str(ckpt),P6_files=weights,policy_or_ROS_execution_started=False,
                P6_state_matches=weights['state.pt']['sha256']=='a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43')
    (out/'inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    shutil.copy2(__file__,out/'collector.py')
    print('GAZEBO_A0_INVENTORY_EXPORTED',len(files),len(world_files),'P6',report['P6_state_matches'])


if __name__=='__main__':
    main()
