cmake_minimum_required(VERSION 3.5)
project(paper_super_sync)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -fPIC")

# Must match the official rog_map/super_planner compile-time semantics so
# header-only code is consistent across translation units.
add_definitions(-DORIGIN_AT_CORNER)
add_definitions(-DFMT_HEADER_ONLY)
add_definitions(-DUSE_ROS1)
# fsm.h expands ROOT_DIR inside its log-path macros; point them at this
# adapter package so runtime logs never land in the official tree.
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_generation
  super_planner
  rog_map
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

add_service_files(FILES EvaluationStep.srv)
generate_messages(DEPENDENCIES std_msgs geometry_msgs nav_msgs sensor_msgs)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs nav_msgs sensor_msgs message_runtime
)

# Official super_planner exports neither INCLUDE_DIRS nor SOURCE_DIR
# (upstream catkin_package lists only LIBRARIES super_planner), so locate the
# pinned official tree from this package's real source path. The layout
# (third_party/paper_super_sync next to third_party/super_official_2ad3419)
# is fixed by the workspace setup and recorded in the run identity.
get_filename_component(_PAPER_SYNC_REAL ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
set(SUPER_OFFICIAL_DIR ${_PAPER_SYNC_REAL}/../super_official_2ad3419)
if(NOT EXISTS ${SUPER_OFFICIAL_DIR}/super_planner/include/fsm/fsm.h)
  message(FATAL_ERROR "official SUPER tree not found next to paper_super_sync: ${SUPER_OFFICIAL_DIR}")
endif()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${SUPER_OFFICIAL_DIR}/super_planner/include
  ${SUPER_OFFICIAL_DIR}/rog_map/include
)

add_executable(paper_super_sync_node src/sync_node.cpp)
add_dependencies(paper_super_sync_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(paper_super_sync_node
  ${catkin_LIBRARIES}
  ${super_planner_LIBRARIES}
  ${rog_map_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
  dw
)
