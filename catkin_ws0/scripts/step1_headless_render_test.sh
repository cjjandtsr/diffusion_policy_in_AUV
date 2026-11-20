#!/bin/bash
# ============================================================================
# Phase 1b Step 1: headless 离屏渲染 5 分钟预筛
# ============================================================================
# 依据: Phase1b_Gym_Wrapper_Design v3.1 §2 (首要验证项, 阻塞性)
# 目的: 5 分钟内确认 4090 headless 镜像下 Gazebo 深度相机能否输出点云
#
# 测试三方案优先级: EGL > xvfb > llvmpipe
# 通过条件:
#   1. ~pointcloud 话题非空, 点数 30k-60k (与 BC 同量级)
#   2. 单步仿真 wall-clock < 1s
#   3. GPU 显存占用 < 500MB/实例
#
# 用法:
#   bash step1_headless_render_test.sh egl       # 测 EGL
#   bash step1_headless_render_test.sh xvfb      # 测 xvfb
#   bash step1_headless_render_test.sh llvmpipe  # 测软件渲染兜底
#   bash step1_headless_render_test.sh prescreen # Zread 建议的 5 分钟预筛
# ============================================================================

set -e
MODE=${1:-prescreen}
WORLD=${WORLD:-${HOME}/AUV/catkin_ws0/src/ego_planner/launch/env_014_obs14.world}
TEST_DURATION=${TEST_DURATION:-30}   # 秒, 仿真时长用于点云采集

# 工作目录
WORK_DIR=$(dirname "$(readlink -f "$0")")
LOG_DIR=${WORK_DIR}/headless_test_logs
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Phase 1b Step 1: headless 离屏渲染验证  mode=${MODE}"
echo "  world: ${WORLD}"
echo "  log:   ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════════"

# 检查 world 文件
if [ ! -f "${WORLD}" ]; then
    echo "[ERR] world 文件不存在: ${WORLD}"
    echo "      请通过 WORLD=<path> 指定, 或修改脚本默认路径"
    exit 1
fi

# ============================================================================
# 模式 1: Zread 建议的 5 分钟预筛 (先做)
# 检查当前 Gazebo 默认能用什么后端, 不实际跑仿真
# ============================================================================
if [ "${MODE}" = "prescreen" ]; then
    echo ""
    echo "[预筛 1/3] 检查 DISPLAY 环境变量..."
    echo "  DISPLAY = '${DISPLAY:-<unset>}'"
    if [ -z "${DISPLAY}" ]; then
        echo "  → 确认: headless 模式 (无 DISPLAY)"
    else
        echo "  → 注意: DISPLAY 已设, 可能不是纯 headless"
    fi

    echo ""
    echo "[预筛 2/3] 启动 gzserver --verbose, 看 Ogre 后端类型 (5 秒后 kill)..."
    PRESCREEN_LOG=${LOG_DIR}/prescreen_$(date +%H%M%S).log
    timeout 5 gzserver --verbose "${WORLD}" > "${PRESCREEN_LOG}" 2>&1 || true
    echo "  log: ${PRESCREEN_LOG}"
    echo ""
    echo "  --- Ogre / RenderSystem 相关行 ---"
    grep -iE "ogre|rendersystem|opengl|egl|xvfb|llvmpipe|render system|gpu" "${PRESCREEN_LOG}" \
        | head -20 || echo "  (无相关日志, 可能 gzserver 启动失败)"

    echo ""
    echo "[预筛 3/3] 检查 GPU 与 EGL 支持..."
    if command -v nvidia-smi &> /dev/null; then
        echo "  nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "  显存:       $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
    else
        echo "  [WARN] nvidia-smi 不可用"
    fi
    if [ -e /usr/lib/x86_64-linux-gnu/libEGL.so ] || [ -e /usr/lib/libEGL.so ]; then
        echo "  EGL lib: 存在"
    else
        echo "  EGL lib: 未找到 (可能需要 apt install libegl1)"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  预筛完成. 根据日志判断:"
    echo "    若 Ogre 日志显示 'GL3PlusRenderSystem' + EGL 上下文 → 走 EGL"
    echo "    若显示 'GL Renderer: llvmpipe' → 软件渲染, 跳到 xvfb"
    echo "    若 gzserver 启动失败 (无任何 Ogre 行) → 必须 xvfb"
    echo ""
    echo "  下一步:"
    echo "    bash $0 egl    # 预筛显示 EGL 可用时"
    echo "    bash $0 xvfb   # 预筛不确定或 EGL 失败时, 直接上 xvfb"
    echo "════════════════════════════════════════════════════════════════"
    exit 0
fi

# ============================================================================
# 完整测试 (egl / xvfb / llvmpipe)
# ============================================================================
TEST_LOG=${LOG_DIR}/${MODE}_$(date +%H%M%S).log
PC_DUMP=${LOG_DIR}/${MODE}_pc_stats.txt

case "${MODE}" in
    egl)
        echo ""
        echo "[EGL 测试] 设置 OGRE 后端 + 启动 gzserver..."
        export OGRE_RTT_MODE=Copy             # EGL 下 Ogre 渲染目标设置
        # Gazebo Classic 不直接支持 EGL flag, 依赖 Ogre 编译时的 EGL plugin
        # 若 prescreen 日志显示 EGL 可用, 直接启动即可
        unset DISPLAY  # 强制无 X
        LAUNCH_CMD="gzserver --verbose ${WORLD}"
        ;;
    xvfb)
        echo ""
        echo "[xvfb 测试] 启动虚拟 X server + gzserver..."
        if ! command -v xvfb-run &> /dev/null; then
            echo "[ERR] xvfb-run 未安装"
            echo "  apt install -y xvfb"
            exit 1
        fi
        LAUNCH_CMD="xvfb-run -a -s '-screen 0 1024x768x24' gzserver --verbose ${WORLD}"
        ;;
    llvmpipe)
        echo ""
        echo "[llvmpipe 测试] 软件渲染兜底..."
        export LIBGL_ALWAYS_SOFTWARE=1
        export GALLIUM_DRIVER=llvmpipe
        if ! command -v xvfb-run &> /dev/null; then
            echo "[ERR] xvfb-run 未安装 (llvmpipe 仍需 X 上下文)"
            exit 1
        fi
        LAUNCH_CMD="xvfb-run -a gzserver --verbose ${WORLD}"
        ;;
    *)
        echo "[ERR] 未知 mode: ${MODE}"
        echo "  可选: prescreen | egl | xvfb | llvmpipe"
        exit 1
        ;;
esac

# 启动 gzserver
echo "  CMD: ${LAUNCH_CMD}"
echo "  log: ${TEST_LOG}"
eval "${LAUNCH_CMD}" > "${TEST_LOG}" 2>&1 &
GZSERVER_PID=$!
echo "  gzserver PID: ${GZSERVER_PID}"

# 等仿真初始化
sleep 5
if ! kill -0 ${GZSERVER_PID} 2>/dev/null; then
    echo "[ERR] gzserver 启动失败, 查看 log: ${TEST_LOG}"
    tail -30 "${TEST_LOG}"
    exit 1
fi

# ============================================================================
# 测点云话题
# ============================================================================
echo ""
echo "[点云测试] 监听 ~pointcloud 话题 ${TEST_DURATION}s, 统计点数..."

# 检查 rostopic 是否可用 (需要先 source ROS)
if ! command -v rostopic &> /dev/null; then
    echo "[ERR] rostopic 不可用, 请先 'source /opt/ros/noetic/setup.bash'"
    kill ${GZSERVER_PID}
    exit 1
fi

# 找点云话题 (无人机模型/插件可能命名不同, 列所有 PointCloud2 话题)
PC_TOPICS=$(rostopic list 2>/dev/null | xargs -I{} sh -c 'rostopic type {} 2>/dev/null | grep -q PointCloud2 && echo {}')
if [ -z "${PC_TOPICS}" ]; then
    echo "[WARN] 未发现 PointCloud2 话题. 可能 world 不含相机模型, 或插件未加载"
    echo "  当前所有话题:"
    rostopic list 2>/dev/null | head -20
    echo "  → 此测试需要含深度相机的 world (env_014_obs14 + 无人机模型)"
    echo "  → 仅 gzserver + world 不含相机, 需配合 spawn drone 或测专用 world"
    kill ${GZSERVER_PID}
    echo "  [SKIP] 点云测试跳过. 但 gzserver 渲染后端启动成功 (见 ${TEST_LOG})"
    echo "  实现 GazeboUAVEnv 时再用完整 launch (PX4 SITL + drone + 相机) 复测"
    exit 0
fi

echo "  发现 PointCloud2 话题:"
echo "${PC_TOPICS}" | sed 's/^/    /'
PC_TOPIC=$(echo "${PC_TOPICS}" | head -1)
echo "  采样 ${PC_TOPIC} ${TEST_DURATION}s..."

# 用 rostopic hz + python 解析点数
timeout ${TEST_DURATION} rostopic echo -n 10 "${PC_TOPIC}" 2>/dev/null \
    | grep -E "^width|^height|^point_step" \
    > "${PC_DUMP}" || true

if [ -s "${PC_DUMP}" ]; then
    echo "  采样 (前 10 帧 width/height/point_step):"
    head -30 "${PC_DUMP}" | sed 's/^/    /'
    # 估算总点数 = width * height (organized) 或仅 width (unorganized)
    WIDTHS=$(grep "^width:" "${PC_DUMP}" | awk '{print $2}' | head -5)
    HEIGHTS=$(grep "^height:" "${PC_DUMP}" | awk '{print $2}' | head -5)
    echo "  width samples: $(echo ${WIDTHS} | tr '\n' ' ')"
    echo "  height samples: $(echo ${HEIGHTS} | tr '\n' ' ')"
else
    echo "  [WARN] 未收到点云数据, 话题存在但无消息"
fi

# ============================================================================
# 测 GPU 显存
# ============================================================================
echo ""
echo "[GPU 显存测试] 当前 gzserver 实例显存占用..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
        | grep -E "gzserver|gazebo|${GZSERVER_PID}" \
        | sed 's/^/  /' \
        || echo "  (gzserver 未占用 GPU 显存, 可能是软件渲染)"
fi

# ============================================================================
# 清理
# ============================================================================
echo ""
echo "[清理] kill gzserver (PID ${GZSERVER_PID})..."
kill ${GZSERVER_PID} 2>/dev/null || true
wait ${GZSERVER_PID} 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  测试完成. 结果文件:"
echo "    ${TEST_LOG}"
echo "    ${PC_DUMP}"
echo ""
echo "  验收 (v3.1 §2.3):"
echo "    [ ] 点云点数 30k-60k 量级 (与 BC 同量级)"
echo "    [ ] 单步 0.3s 仿真 wall-clock < 1s (需 step2 完整 env 测)"
echo "    [ ] GPU 显存 < 500MB/实例"
echo ""
echo "  若本次测试通过 → 记录 mode=${MODE}, 进入 step 2 (StepDriver 抽象)"
echo "  若失败 (点云空/无 PC 话题/gzserver 死) → 试下一个 mode"
echo "════════════════════════════════════════════════════════════════"
