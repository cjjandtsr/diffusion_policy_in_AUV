# PX4 local_planner 运动学桥接

修订v1_1（2026-11-10）：调用官方默认参数后，通过桥接launch将原生巡航／上下速度设为0.5m/s、水平／上下加速度参数设为0.25m/s²。该加速度参数参与官方传感器距离限速公式；理想位置航点转换不等价于施加全轨迹加速度硬约束，不宣称与Fast动力学约束等价。ROS编译与真实参数解析待WSL验证。修改前文件保存在`program/code_backups/20261110_desktop_completion/`。

日期2026-11-10。官方库固定 c386852c1b8e2084094e194d9dcd46fdd0c01646。
原算法来自 PX4/PX4-Avoidance，调用 LocalPlanner 与 WaypointGenerator；不自行实现直方图、搜索或避障策略。

官方文本源码通过GitHub连接器读取。原生HTTPS下载三次TLS失败。下载范围为文件树中的文本文件，不含模型图像、网格、PDF、bag；缺失一份仿真灯具SDF，与本桥接运行无关。媒体不参与算法编译。

ROS服务使用同步仿真时间；每回合重建原生算法及航点生成对象，清空历史。仅接收世界系局部传感器点云、当前状态、目标。未提供障碍真值或全局地图。

原生输出含位置航点、速度控制掩码（NaN）与yaw。Python以原生 avoidanceOutput.cruise_velocity 为位置参考推进速度，将当前状态到位置航点作线性时频转换；位置未定义的轴使用原生速度。返回8×11参考并交原H4执行。速度掩码在日志以null与显式mask保留。此结果属于理想位置跟踪运动学适配，不是MAVROS/PX4真实闭环，不宣称控制语义等价。

launch视场默认对应现存DepthCameraIntrinsics的320×240、hfov=1.047198rad；执行前必须核对实际评价传感器，改变视场需同时修改该launch记录身份。原生动态参数使用官方生成配置默认值，PX4固件参数调用官方setDefaultPx4Parameters，原生smoothing保留。

偏航与平移独立转换：有效非零原生yaw_rate按速率推进至原生yaw目标，其余情况直接跟踪原生yaw目标；原地转向不会因平移距离为零冻结。

未在Windows编译ROS C++，未运行端到端冒烟或正式实验。Linux编译、传感器视场对应和实际有效输出是待验证项。
