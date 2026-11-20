cmake_minimum_required(VERSION 3.0.2)
project(paper_gazebo_planners)
find_package(catkin REQUIRED)
catkin_package()

# Every script published by this package, and nothing else: the previous version
# installed ../../scripts/paper_super_legacy_command.py, which no longer exists
# and made the whole workspace unbuildable.
catkin_install_python(PROGRAMS
  scripts/local_cloud_bridge.py
  scripts/send_fast_goal.py
  scripts/send_super_goal.py
  scripts/paper_cmd_relay.py
  scripts/paper_super_command_relay.py
  scripts/paper_fast_command_relay.py
  scripts/paper_cmd_import.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
