<launch>
  <!-- 
    ══════════════════════════════════════════════════════════════
    ML Planner 部署启动文件
    用训练好的扩散模型替代 EGO-Planner
    ══════════════════════════════════════════════════════════════
    
    文件位置: plan_manage/launch/ml_planner_deploy.launch
    
    使用方法:
    ─────────
    # 终端1: 启动仿真器
    roslaunch px4 fast_racing.launch
    
    # 终端2: 起飞 (完成后Ctrl+C)
    rosrun offboard_pkg offboard_node 30
    
    # 终端3: 启动ML规划器
    roslaunch ego_planner ml_planner_deploy.launch
    
    # 在rviz中点击 "2D Nav Goal" 发送目标点
    
    ══════════════════════════════════════════════════════════════
  -->

  <arg name="drone_id" default="0"/>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- 1. ML规划器节点 (替代 ego_planner_node + traj_server)        -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  
  <node name="drone_$(arg drone_id)_ml_planner" 
        pkg="ego_planner"
        type="ml_planner_node.py"
        output="screen"
        required="true">
  </node>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- 2. PX4 控制器 (必须，负责底层控制)                            -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  
  <include file="$(find px4ctrl)/launch/run_node.launch"/>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- 3. 可视化                                                    -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  
  <!-- RViz -->
  <node pkg="rviz" type="rviz" name="rviz" 
        args="-d $(find ego_planner)/launch/gazebo.rviz"
        output="screen"/>
  
  <!-- 无人机轨迹可视化 -->
  <node pkg="odom_visualization" 
        name="drone_$(arg drone_id)_odom_visualization" 
        type="odom_visualization" 
        output="screen">
    <remap from="~odom" to="/mavros/local_position/odom"/>
    <param name="color/a" value="1.0"/>    
    <param name="color/r" value="0.0"/>        
    <param name="color/g" value="0.0"/>        
    <param name="color/b" value="1.0"/>       
    <param name="covariance_scale" value="100.0"/>       
    <param name="robot_scale" value="1.0"/>
    <param name="tf45" value="false"/>
    <param name="drone_id" value="$(arg drone_id)"/>
  </node>

</launch>
