"""Read frozen task/world evidence and export the paper request's geometry rows."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.workspace), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    test_path = root/'wsl_snapshot/routea_test12_independent_run02/results/bank/bank.json'
    test = read(test_path)
    freeze_path = root/'wsl_snapshot/routea_final_freeze_run02/routea_final_freeze_delivery_run02.tar.gz'
    member = 'routea_final_freeze_run01/bank/bank.json'
    with tarfile.open(freeze_path) as archive:
        ext = json.load(archive.extractfile(member))
        index = json.load(archive.extractfile('routea_final_freeze_run01/O_prime_index.json'))
    reference_path = root/'wsl_snapshot/routea_a2_selective_keep_run01/reference/tasks.json'
    train_worlds = sorted({t['world_id'] for t in read(reference_path)['tasks']})
    asset_root = root/'wsl_snapshot/routea_paper_prereq_run03/evidence/worlds'
    asset_paths = {p.name.removesuffix('.world.obstacles.json'): p for p in asset_root.rglob('*.world.obstacles.json')}
    # Frozen banks carry the exact obstacle primitives used by evaluation.
    worlds = {**test['worlds'], **ext['worlds']}
    rows = []
    for split, ids in [('O_prime', sorted(worlds)), ('train_admitted', train_worlds)]:
        for uid in ids:
            source = asset_paths.get(uid)
            asset = read(source) if source else {}
            geometry = worlds[uid]['geometry'] if split == 'O_prime' else asset['obstacles']
            region = asset.get('obstacle_region', [])
            rows.append(dict(split=split,world_id=uid,
                world_x_min='',world_x_max='',world_y_min='',world_y_max='',
                obstacle_region_json=json.dumps(region),
                box_count=sum(o['shape']=='box' for o in geometry),
                cylinder_count=sum(o['shape']=='cylinder' for o in geometry),
                other_count=sum(o['shape'] not in ('box','cylinder') for o in geometry),
                total_obstacles=len(geometry),source=str(source) if source else 'frozen bank geometry',
                source_sha256=sha(source) if source else '',
                world_bounds_status='not in obstacle JSON; obstacle_region is not world size'))
    task_rows=[]
    for dataset, bank in [('test12',test),('ext_v2',ext)]:
        for task in bank['tasks']:
            payload=task['payload'];start,goal=payload['start_pos'],payload['goal_pos']
            task_rows.append(dict(dataset=dataset,task_id=task['task_id'],world_id=task['world_id'],
                start_x=start[0],start_y=start[1],start_z=start[2],goal_x=goal[0],goal_y=goal[1],goal_z=goal[2],
                straight_distance_3d_m=math.dist(start,goal),straight_distance_xy_m=math.dist(start[:2],goal[:2]),
                source_bank_content_hash=bank['content_hash']))
    write_csv(out/'worlds_63.csv',rows)
    write_csv(out/'tasks_199.csv',task_rows)
    (out/'O_prime_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'geometry_provenance.json').write_text(json.dumps(dict(test_bank_path=str(test_path),
        test_bank_sha256=sha(test_path),extension_archive=str(freeze_path),extension_member=member,
        extension_archive_sha256=sha(freeze_path),reference_tasks=str(reference_path),
        reference_tasks_sha256=sha(reference_path),O_prime_worlds=len(worlds),training_worlds=len(train_worlds),
        tasks=len(task_rows),world_size_missing=True),indent=2)+'\n',encoding='utf-8')
    print('GEOMETRY_EXPORTED',len(worlds),len(train_worlds),len(task_rows))


if __name__=='__main__':
    main()
