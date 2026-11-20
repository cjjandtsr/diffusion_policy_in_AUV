#!/bin/bash

# 设置循环次数 - 增加数据量以获得更丰富的训练数据
MAX_LOOPS=3
DATA_DIR="$HOME/AUV/avoidance_dataset"

# 创建专门的数据目录
mkdir -p $DATA_DIR/rosbags
mkdir -p $DATA_DIR/processed

echo "=== 开始绕障运动数据收集（同步模式 3.3Hz）==="

# 查找现有bag文件的最大序号
MAX_EXISTING_NUM=0
for bag_file in $DATA_DIR/rosbags/avoidance_*.bag; do
    if [ -f "$bag_file" ]; then
        filename=$(basename "$bag_file")
        num=$(echo $filename | sed 's/avoidance_\([0-9]*\)\.bag/\1/')
        if [ "$num" -gt "$MAX_EXISTING_NUM" ] 2>/dev/null; then
            MAX_EXISTING_NUM=$num
        fi
    fi
done

echo "现有最大序号: $MAX_EXISTING_NUM"

# 计算起始序号
START_NUM=$((MAX_EXISTING_NUM + 1))
END_NUM=$((MAX_EXISTING_NUM + MAX_LOOPS))

echo "将收集 $MAX_LOOPS 轮绕障运动数据"
echo "数据保存到: $DATA_DIR"
echo "起始序号: $START_NUM, 结束序号: $END_NUM"

loop_count=$START_NUM
while [ $loop_count -le $END_NUM ]; do
    echo "=== 第 $((loop_count - START_NUM + 1))/$MAX_LOOPS 轮绕障数据收集 ==="
    echo "文件序号: $loop_count"
    
    BAG_NAME="avoidance_${loop_count}.bag"
    
    # 检查文件是否已存在（防止覆盖）
    if [ -f "$DATA_DIR/rosbags/$BAG_NAME" ]; then
        echo "警告: 文件 $BAG_NAME 已存在，跳过此轮..."
        loop_count=$((loop_count + 1))
        continue
    fi
    
    echo "阶段1: 启动offboard控制节点进行起飞..."
    rosrun offboard_pkg offboard_node 30 &
    sleep 40
    pkill -f "offboard_node"

    echo "阶段2: 启动ego规划器系统..."
    roslaunch ego_planner single_run_in_gazebo.launch &
    sleep 15
    roslaunch px4ctrl run_node.launch &
    sleep 10
    roslaunch ego_planner rviz.launch &
    sleep 10

    # 配置PX4Ctrl参数
    echo "配置PX4Ctrl参数..."
    rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool False
    sleep 2
    rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool False
    sleep 2
    rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True
    sleep 3
    rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool True
    sleep 3

    # 阶段3: 启动同步节点和rosbag记录
    echo "=== 启动同步节点 (3.3Hz) ==="
    rosrun ego_planner sync_recorder_node &
    SYNC_PID=$!
    sleep 3

    echo "=== 开始记录同步数据 (/sync_data) ==="
    rosbag record --tcpnodelay /sync_data /current_phase -O $DATA_DIR/rosbags/$BAG_NAME &
    ROSBAG_PID=$!
    sleep 5

    echo "=== 启动绕障目标生成器 ==="
    rosrun custom_target_generator random_target_generator

    echo "=== 绕障数据收集完成，停止记录 ==="

    # 停止rosbag记录和同步节点
    kill $ROSBAG_PID
    kill $SYNC_PID
    sleep 3

    echo "=== 关闭所有节点 ==="

    # 确保无人机完全降落
    echo "等待无人机完全降落..."
    sleep 10
    
    # 关闭所有相关节点
    pkill -f "single_run_in_gazebo.launch"
    pkill -f "run_node.launch" 
    pkill -f "rviz.launch"
    pkill -f "random_target_generator"
    pkill -f "ego_planner_node"
    pkill -f "traj_server"
    pkill -f "odom_visualization"
    pkill -f "px4ctrl_node"
    pkill -f "rviz"
    pkill -f "planner_manager"
    pkill -f "ego_replan_fsm"
    pkill -f "sync_recorder_node"   # 额外确保同步节点被杀死
    
    echo "=== 第 $loop_count/$MAX_LOOPS 轮绕障数据收集完成 ==="
    echo "Bag文件: $DATA_DIR/rosbags/$BAG_NAME"
    
    # 如果不是最后一轮，等待一段时间再开始下一轮
    if [ $loop_count -lt $MAX_LOOPS ]; then
        echo "等待10秒准备下一轮绕障数据收集..."
        sleep 10
    fi
    
    loop_count=$((loop_count + 1))
done

echo "=== 所有 $MAX_LOOPS 轮绕障数据收集已完成 ==="
echo "rosbag数据保存在: $DATA_DIR/rosbags/"
echo "文件列表:"
ls -lh $DATA_DIR/rosbags/*.bag
echo "=== 绕障数据收集完成 ==="