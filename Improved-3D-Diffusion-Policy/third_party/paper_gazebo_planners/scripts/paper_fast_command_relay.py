#!/usr/bin/env python3
"""Fast-Planner kino entry point of the generic command bridge.

Fast's traj_server publishes its own quadrotor_msgs/PositionCommand
(md5 4712f0609ca29a79af79a35ca3e3967a), which px4ctrl cannot consume. The launch
retargets traj_server to /paper/fast/pos_cmd and this relay forwards it.
"""
import os
import sys

# roslaunch executes the catkin stub in devel/lib/<pkg>/, so sys.path[0] is that
# directory, which contains a stub of the same name; importing it does not expose
# the real module's names ("ImportError: cannot import name 'run'"). The stub sets
# __file__ to the source path, so prepending the real scripts dir fixes the import.
def _front(path):
    """Move `path` to sys.path[0]; a plain insert is skipped when PYTHONPATH already
    contains it, which is exactly the case that made the md5 guard fail."""
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
_front(_SCRIPTS_DIR)

from paper_cmd_relay import run

if __name__ == '__main__':
    sys.exit(run('/paper/fast/pos_cmd', node_name='paper_fast_command_relay'))
