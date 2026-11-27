"""O7（只读）：示范的反应曲线——前方障碍越近，示范偏离目标方位越多吗？

数据：merged_body_12.zarr
  data/state  (N, 9)：pos_world(3) + goal_body(3) + dist(1) + sin_yaw + cos_yaw
  data/action (N, 11)：不使用
  data/point_cloud (N, 4096, 6)：xyz_body(3) + color(3)
  meta/episode_ends
口径（与 P6 的 a15 尽量对齐，差别见文末）：
  - 目标方位：goal_body 的水平方位角 θg（机体系，x 前 y 左）。只统计 |θg| ≤ 30°（目标方位在水平视场内）、离目标 ≥ 2 m 的帧。
  - 前方自由距离 F：机体系点云中，去掉全零补位点与世界高度 < 0.15 m 的地面点；取与机体同高 ±0.3 m、
    到"沿目标方位的水平射线"的横向距离 < 0.3 m、沿射线距离 > 0.1 m 的点，F = 这些点沿射线距离的最小值；没有这样的点记为 ≥ 10 m。
  - 偏角：该帧之后 4 帧（1.2 s）的世界系位移，用该帧偏航转到机体系，取其水平方向与 θg 的夹角（0—180°）。
  - 速度：上述位移长度 / 1.2 s。
用法：
  python3 -I o7_demo_reaction.py --selftest
  python3 -I o7_demo_reaction.py <merged_body_12.zarr 路径> <输出目录>
输出：<输出目录>/o7_demo_reaction.json、o7_demo_reaction.txt"""
import json, math, os, sys
import numpy as np

H_BAND, LAT, Z_GROUND, FOV, K = 0.3, 0.3, 0.15, 30.0, 4
BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 99)]


def free_distance(pc_xyz, theta, pos_z):
    """pc_xyz: (n,3) 机体系；theta: 目标方位（弧度）；pos_z: 世界高度"""
    p = pc_xyz[np.any(pc_xyz != 0, axis=1)]
    p = p[(p[:, 2] + pos_z >= Z_GROUND) & (np.abs(p[:, 2]) <= H_BAND)]
    if len(p) == 0:
        return 10.0
    u = np.array([math.cos(theta), math.sin(theta)])
    s = p[:, :2] @ u
    lat = np.abs(p[:, 0] * u[1] - p[:, 1] * u[0])
    m = (s > 0.1) & (lat < LAT)
    return float(min(s[m].min(), 10.0)) if m.any() else 10.0


def deviation(pos_t, pos_tk, yaw, theta):
    d = pos_tk - pos_t
    c, s = math.cos(-yaw), math.sin(-yaw)
    bx, by = c * d[0] - s * d[1], s * d[0] + c * d[1]
    if math.hypot(bx, by) < 1e-6:
        return float('nan'), 0.0
    ang = abs((math.degrees(math.atan2(by, bx) - theta) + 180) % 360 - 180)
    return ang, float(np.linalg.norm(d)) / (0.3 * K)


def selftest():
    pc = np.zeros((10, 3)); pc[0] = [3.0, 0.1, 0.0]; pc[1] = [2.0, 1.0, 0.0]; pc[2] = [1.0, 0.0, -1.9]
    f = free_distance(pc, 0.0, 2.0)
    assert abs(f - 3.0) < 1e-9, f                      # 第 2 点横向 1.0 m 不计；第 3 点为地面（世界高 0.1）不计
    a, v = deviation(np.zeros(3), np.array([1.0, 1.0, 0.0]), 0.0, 0.0)
    assert abs(a - 45.0) < 1e-9 and abs(v - math.sqrt(2) / 1.2) < 1e-9, (a, v)
    a2, _ = deviation(np.zeros(3), np.array([0.0, 1.0, 0.0]), math.pi / 2, 0.0)  # 偏航 90°，世界 +y 即机体前方
    assert abs(a2) < 1e-9, a2
    print('selftest ok: F=3.0, 偏角 45°, 偏航换算 0°')


def main(zpath, out):
    import zarr
    g = zarr.open(zpath, 'r')
    S = g['data/state'][:]; ends = g['meta/episode_ends'][:]; PC = g['data/point_cloud']
    N = len(S); starts = np.concatenate([[0], ends[:-1]])
    ep_end = np.empty(N, int)
    for a, b in zip(starts, ends): ep_end[a:b] = b
    rows = []
    CH = 2000
    for c0 in range(0, N, CH):
        pcs = PC[c0:min(N, c0 + CH), :, :3]
        for i in range(len(pcs)):
            t = c0 + i
            if t + K >= ep_end[t] or S[t, 6] < 2.0:
                continue
            theta = math.atan2(S[t, 4], S[t, 3])
            if abs(math.degrees(theta)) > FOV:
                continue
            yaw = math.atan2(S[t, 7], S[t, 8])
            F = free_distance(pcs[i], theta, float(S[t, 2]))
            ang, v = deviation(S[t, :3].astype(float), S[t + K, :3].astype(float), yaw, theta)
            rows.append((F, ang, v))
    R = np.array(rows)
    res = {'n_frames_total': int(N), 'n_used': int(len(R)), 'bins': []}
    lines = ['示范反应曲线（merged_body_12，目标方位在视场内、离目标 ≥ 2 m 的帧）', '前方自由距离 m | 帧数 | 偏角中位 / 90 分位 ° | 速度中位 m/s']
    for lo, hi in BINS:
        m = (R[:, 0] >= lo) & (R[:, 0] < hi) & np.isfinite(R[:, 1])
        if m.sum() == 0:
            continue
        b = dict(lo=lo, hi=hi, n=int(m.sum()), ang_med=float(np.median(R[m, 1])), ang_p90=float(np.percentile(R[m, 1], 90)), v_med=float(np.median(R[m, 2])))
        res['bins'].append(b)
        lines.append('%s—%s | %d | %.1f / %.1f | %.2f' % (lo, '∞' if hi == 99 else hi, b['n'], b['ang_med'], b['ang_p90'], b['v_med']))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'o7_demo_reaction.json'), 'w'), ensure_ascii=False, indent=1)
    open(os.path.join(out, 'o7_demo_reaction.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        main(sys.argv[1], sys.argv[2])
# 与 P6 a15 的差别：a15 用几何真值与"规划终点方向"；这里只有点云（受视场与遮挡限制）与"之后 1.2 s 的实际位移"。
# Claude 会用 P6 原迹按本脚本同一口径（执行位移、点云近似由几何替代）另算一版对照。
