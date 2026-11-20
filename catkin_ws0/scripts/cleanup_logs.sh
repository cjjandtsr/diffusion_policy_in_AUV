#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 定时清理日志缓存 (cron hook)
# 保留最近 N 天的日志，清理过期文件
# ══════════════════════════════════════════════════════════════

set -e

KEEP_DAYS="${KEEP_DAYS:-2}"
DRY_RUN="${DRY_RUN:-false}"
LOG_DIR="${AUV_LOG_DIR:-/root/autodl-tmp/logs}"
LOG_FILE="$LOG_DIR/log_cleanup.log"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

clean_old_dirs() {
    local base="$1"
    local label="$2"
    if [ ! -d "$base" ]; then
        log "SKIP $label: $base 不存在"
        return
    fi
    log "扫描 $label: $base"
    find "$base" -maxdepth 1 -type d -mtime +"$KEEP_DAYS" 2>/dev/null | while read -r d; do
        # 跳过自身
        [ "$d" = "$base" ] && continue
        local size=$(du -sh "$d" 2>/dev/null | cut -f1)
        log "  删除 [$size] $d"
        if [ "$DRY_RUN" != "true" ]; then
            rm -rf "$d"
        fi
    done
}

clean_old_files() {
    local dir="$1"
    local pattern="$2"
    local label="$3"
    if [ ! -d "$dir" ]; then
        log "SKIP $label: $dir 不存在"
        return
    fi
    log "扫描 $label: $dir/$pattern"
    find "$dir" -maxdepth 1 -name "$pattern" -type f -mtime +"$KEEP_DAYS" 2>/dev/null | while read -r f; do
        local size=$(du -sh "$f" 2>/dev/null | cut -f1)
        log "  删除 [$size] $f"
        if [ "$DRY_RUN" != "true" ]; then
            rm -f "$f"
        fi
    done
}

truncate_large_logs() {
    local max_mb="${1:-500}"
    local pattern="$2"
    log "截断大日志 (>${max_mb}MB): $pattern"
    find "$LOG_DIR" -maxdepth 1 -name "$pattern" -type f -size +"${max_mb}M" 2>/dev/null | while read -r f; do
        local size=$(du -sh "$f" 2>/dev/null | cut -f1)
        log "  截断 [$size] $f → 保留最后 100MB"
        if [ "$DRY_RUN" != "true" ]; then
            tail -c 100M "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
        fi
    done
}

log "════════ 日志清理开始 (保留 ${KEEP_DAYS} 天) ════════"

# 1. ROS 日志 (~/.ros/log)
clean_old_dirs "$HOME/.ros/log" "ROS日志"

# 2. PX4 SITL 飞行日志
clean_old_dirs "$HOME/AUV/PX4-Autopilot/build/px4_sitl_default/log" "PX4-ulg"
clean_old_dirs "$HOME/AUV/PX4-Autopilot/build/px4_sitl_default/tmp/rootfs/log" "PX4-rootfs-log"

# 3. 运行日志 (> 500MB 的 sitl.log 只保留尾部)
truncate_large_logs 500 "sitl*.log"

# 4. 过期日志文件 (仅匹配已知日志模式)
clean_old_files "$LOG_DIR" "sitl*.log" "run-sitl"
clean_old_files "$LOG_DIR" "planner*.log" "run-planner"
clean_old_files "$LOG_DIR" "fast_test*.log" "run-fast_test"
clean_old_files "$LOG_DIR" "gdb-server-console-*.log" "run-gdb"
clean_old_files "$LOG_DIR" "log_cleanup.log" "run-cleanup-self"

log "════════ 清理完成 ════════"
