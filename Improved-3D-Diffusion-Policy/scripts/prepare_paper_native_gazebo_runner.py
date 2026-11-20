"""Extend an execution-side task flight runner, preserving its monitor and takeoff."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    source = a.runner.read_text(encoding='utf-8')
    marker = 'if [ "$ENGINE" = "ego" ]; then\n'
    start = source.index(marker, source.index('PLANNER_LAUNCH='))
    addition = '''if [ "$ENGINE" = "fast" ] || [ "$ENGINE" = "super" ]; then
    if [ "$ENGINE" = "super" ]; then
        python3 "$LEGACY_COMMAND_SCRIPT" &
    fi
    # Restore the planner overlay after the original runner sources the controller workspace.
    set +u
    source "$NATIVE_SETUP"
    set -u
    # NATIVE_LAUNCH is a complete roslaunch argument vector in a JSON file.
    # Spawn via Python argv, not eval; launch inherits the execution-side ROS overlay.
    python3 -c 'import json,os,sys; os.execvp("roslaunch", ["roslaunch"]+json.load(open(sys.argv[1])))' "$NATIVE_LAUNCH_JSON" &
    PLANNER_PID=$!
    sleep 10
elif [ "$ENGINE" = "ego" ]; then
'''
    source = source[:start] + source[start:].replace(marker, addition, 1)
    # A planner baseline does not need a learning checkpoint.
    source = source.replace('[ -f "$CKPT" ]       ||',
                            '[ "$ENGINE" != "bc" ] || [ -f "$CKPT" ] ||')
    # Existing runner owns platform startup and cleanup; add only our node families.
    marker = '    log "清理所有进程..."'
    if marker in source:
        source = source.replace(marker, marker + '''
    rosnode kill /paper_local_cloud_bridge /paper_fast_goal /paper_fast_traj_server /fast_planner_node /paper_super /paper_super_goal /paper_super_command_relay /paper_super_legacy_command 2>/dev/null || true''', 1)
    with a.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)
    a.output.with_suffix('.identity.json').write_text(json.dumps({
        'source': str(a.runner), 'source_sha256': hashlib.sha256(a.runner.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
        'scope': 'planner launch dispatch and checkpoint prerequisite only; original task monitor retained',
    }, indent=2), encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
