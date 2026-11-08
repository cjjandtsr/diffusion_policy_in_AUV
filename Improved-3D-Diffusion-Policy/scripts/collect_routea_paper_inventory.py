"""Export H3/H10 and Gazebo A.0 filesystem evidence, without starting ROS or loading models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def is_evidence_export(path):
    return any(part.startswith('routea_paper_prereq_run') for part in Path(path).parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--auv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo, auv, out = Path(args.repo).resolve(), Path(args.auv).resolve(), Path(args.output).resolve()
    out.relative_to(auv / 'program' / 'results')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='routea_paper_inventory_v1', repo=str(repo), checkpoints={},
                  native_sources={}, world_assets=[], copied_evidence=[], commands={})
    ckroot = repo / 'data/outputs/dppo_uav_rebuild'
    paths = dict(P3='routea_common_b2560_seed202609203_v1/checkpoint_update_000256',
                 P5='routea_xk_run01/main/K/checkpoint_update_000256',
                 P6_u192='routea_a2_selective_keep_run01/T/checkpoint_update_000192',
                 P9='routea_f5_run01/P9/checkpoint_update_000192',
                 P11='routea_f5_run01/P11/checkpoint_update_000192')
    for name, relative in paths.items():
        folder = ckroot / relative
        files = {}
        for filename in ('header.json', 'SHA256.json', 'COMMITTED.json', 'state.pt'):
            path = folder / filename
            if path.is_file():
                files[filename] = dict(bytes=path.stat().st_size, sha256=sha(path))
                if filename != 'state.pt':
                    destination = out / 'checkpoints' / name / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
        report['checkpoints'][name] = dict(path=str(folder), files=files)
    results = auv / 'program/results'
    for source in [repo] + sorted(results.glob('*/native*')):
        if source.is_dir():
            report['native_sources'][str(source)] = {
                str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
                if p.is_file() and p.suffix in ('.py', '.yaml', '.patch')
                and '__pycache__' not in p.parts}
    evidence = list((repo / 'data/eval_configs').rglob('*.json'))
    evidence += list((repo / 'data').glob('*admission*.json'))
    for root in (results, auv / 'program/jindu'):
        for pattern in ('bank.json', '*admission*.json', '*exclusion*.json', '*plan*.json'):
            evidence.extend(p for p in root.rglob(pattern)
                            if out not in p.parents and not is_evidence_export(p))
    for index, path in enumerate(sorted(set(evidence))):
        destination = out / 'history' / ('%05d_' % index + path.name)
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
        report['copied_evidence'].append(dict(source=str(path), copy=str(destination.relative_to(out)), sha256=sha(path)))
    worlds = auv / 'PX4-Autopilot/Tools/sitl_gazebo/worlds'
    for split in ('train', 'val', 'test'):
        for path in sorted((worlds / split).glob('*.obstacles.json')):
            destination = out / 'worlds' / split / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            report['world_assets'].append(dict(split=split, source=str(path),
                copy=str(destination.relative_to(out)), sha256=sha(path)))
    catkin = auv / 'catkin_ws0'
    for directory, pattern in ((catkin / 'scripts', '*.sh'), (catkin / 'src', 'package.xml'),
                               (catkin / 'src', '*.launch'), (catkin / 'src', '*.yaml'),
                               (catkin / 'src/px4ctrl', '*.cpp')):
        for path in sorted(directory.rglob(pattern)):
            destination = out / 'ros_sources' / path.relative_to(catkin)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    model_root = auv / 'PX4-Autopilot/Tools/sitl_gazebo/models'
    for path in sorted(model_root.rglob('*')):
        if path.is_file() and path.suffix in ('.sdf', '.xacro', '.urdf'):
            destination = out / 'sensor_models' / path.relative_to(model_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for name, command in dict(gpu=['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv'],
                              ros=['rosversion', '-d'], gazebo=['gazebo', '--version'],
                              px4=['git', '-C', str(auv / 'PX4-Autopilot'), 'rev-parse', 'HEAD']).items():
        if shutil.which(command[0]):
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            report['commands'][name] = dict(argv=command, rc=result.returncode,
                                            stdout=result.stdout, stderr=result.stderr)
        else:
            report['commands'][name] = dict(argv=command, missing=True)
    report['limitations'] = ['Filesystem search cannot prove never-used status; Claude must review history coverage.',
                            'No ROS launch, controller tracking, world loading or checkpoint load performed.',
                            'Checkpoint paths are expected historical locations; missing entries are reported, not repaired.']
    with (out / 'inventory.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    sums = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    with (out / 'SHA256.json').open('x', encoding='utf-8') as stream:
        json.dump(sums, stream, indent=2)
    print('PAPER_INVENTORY_EXPORTED', out, len(sums), 'not an H3/H10 acceptance')


if __name__ == '__main__':
    main()
