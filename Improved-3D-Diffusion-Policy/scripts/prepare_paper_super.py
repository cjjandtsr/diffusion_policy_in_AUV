"""Fetch pinned official SUPER and derive a ROS1 sensor/vehicle configuration."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

COMMIT = '2ad3419c127a617c6d7df6925e81a14175a9c096'


def fetch(destination):
    destination.mkdir(parents=True, exist_ok=False)
    url = 'https://codeload.github.com/hku-mars/SUPER/tar.gz/' + COMMIT
    data = urllib.request.urlopen(url, timeout=120).read()
    files, omitted = {}, []
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for item in archive:
            if not item.isfile():
                continue
            relative = Path(*Path(item.name).parts[1:])
            # Demo maps, movies and meshes are not used by the real sensor adapter.
            if relative.suffix.lower() in {'.pcd', '.dae', '.gif', '.mp4', '.pdf'}:
                omitted.append(str(relative))
                continue
            content = archive.extractfile(item).read()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            files[str(relative)] = hashlib.sha256(content).hexdigest()
    (destination / 'PAPER_DOWNLOAD_MANIFEST.json').write_text(json.dumps(dict(
        upstream='https://github.com/hku-mars/SUPER', commit=COMMIT,
        archive_sha256=hashlib.sha256(data).hexdigest(), files=files,
        omitted_demo_assets=omitted), indent=2), encoding='utf-8')


def configure(args):
    import yaml
    source = args.source / 'super_planner/config/click_smooth_ros1.yaml'
    cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
    cfg['fsm'].update(click_goal_topic=args.goal_topic, click_height=-10.,
                      cmd_topic='/paper_super/position_cmd',
                      mpc_cmd_topic='/paper_super/polynomial', timer_en=True)
    cfg['super_planner'].update(robot_r=args.radius, goal_yaw_en=False)
    cfg['traj_opt']['boundary'].update(max_vel=args.max_velocity, max_acc=args.max_acceleration)
    mapping = cfg['rog_map']
    mapping.update(load_pcd_en=False, virtual_ground_height=args.ground,
                   virtual_ceil_height=args.ceiling)
    mapping['ros_callback'].update(enable=True, cloud_topic=args.cloud_topic,
                                  odom_topic=args.odom_topic)
    mapping['raycasting'].update(enable=True, ray_range=[args.min_range, args.max_range])
    # The source demo assumes LiDAR and has only six metres of vertical map.
    mapping['map_size'][2] = args.map_height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('fetch')
    download.add_argument('--destination', type=Path, required=True)
    config = commands.add_parser('config')
    config.add_argument('--source', type=Path, required=True)
    config.add_argument('--output', type=Path, required=True)
    for name in ['cloud-topic', 'odom-topic', 'goal-topic']:
        config.add_argument('--' + name, required=True)
    for name in ['radius', 'max-velocity', 'max-acceleration', 'ground', 'ceiling',
                 'map-height', 'min-range', 'max-range']:
        config.add_argument('--' + name, type=float, required=True)
    args = parser.parse_args()
    fetch(args.destination) if args.command == 'fetch' else configure(args)


if __name__ == '__main__':
    main()
