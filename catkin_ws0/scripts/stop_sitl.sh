#!/bin/bash
echo "[stop_sitl] 清理 SITL 进程..."

pkill -9 -f "fast_test.launch"          2>/dev/null
pkill -9 -f "px4_sitl_default/bin/px4"  2>/dev/null
pkill -9 -f "gzserver"                  2>/dev/null
pkill -9 -f "gzclient"                  2>/dev/null
pkill -9 -f "mavros_node"               2>/dev/null
sleep 2

# 确认清理
remaining=$(ps aux | grep -E "px4_sitl|gzserver|gzclient|mavros|fast_test" | grep -v grep | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "[stop_sitl] 全部已停止"
else
    echo "[stop_sitl] 仍有 $remaining 个残留进程，手动检查:"
    ps aux | grep -E "px4_sitl|gzserver|gzclient|mavros|fast_test" | grep -v grep
fi
