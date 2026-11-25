"""只读：33 个直线被挡任务，能否用最短路线上的 1 个或 2 个中间点把路线拆成"每段直线都可通过"的几段。
用法：python3 -I subgoal_feasibility.py <项目仓库根目录> <blocked_tasks_dev420.json> <geo_field.py 所在目录>
输出 subgoals_dev420_blocked.json：每个任务的中间点列表（世界坐标，按经过顺序；1 个中间点时取总路程最短者，2 个时取总路程最短的一对）。
直线可通过 = 该段直线上最小净空 ≥ 0.3 m（0.3 m 粗栅格近似）。中间点取自净空 ≥ 0.3 m 的测地最短路线上的点。"""
import ast, glob, json, os, sys, numpy as np
# 2026-11-17：项目仓库整理后，这些原迹改为精简的 *.jsonl.gz（所用字段均保留）；本脚本同时接受 *.jsonl 与 *.jsonl.gz。
import gzip as _gz
_topen = lambda f: _gz.open(f, 'rt') if f.endswith('.gz') else open(f)
repo, blkf, gdir = sys.argv[1:4]; sys.path.insert(0, gdir)
from geo_field import World, line_clear
BLK = set(json.load(open(blkf))); tasks = {}
for f in glob.glob(f'{repo}/jindu/routea_f5_run01/results/eval_dev/full/P6/traces/*.jsonl*'):
    s = json.loads(_topen(f).readline())
    if s['task_id'] in BLK: tasks.setdefault(s['task_id'], (s['task'] if isinstance(s['task'], dict) else ast.literal_eval(s['task']), s['geometry']))
res = []
for tid, (task, geom) in sorted(tasks.items()):
    s0, g = np.array(task['start_pos']), np.array(task['goal_pos']); lo = np.minimum(s0, g) - 6; hi = np.maximum(s0, g) + 6; lo[2], hi[2] = task['z_range']
    W = World(geom, lo, hi); path = W.field(g).path(s0, 1e9); ok = lambda a, b: line_clear(W, a, b) >= .3
    one = [w for w in path if ok(s0, w) and ok(w, g)]
    n = 1 if one else None; legs = None; wp = None
    if one:
        w = min(one, key=lambda w: np.linalg.norm(w - s0) + np.linalg.norm(g - w)); legs = (float(np.linalg.norm(w - s0)), float(np.linalg.norm(g - w))); wz = float(w[2]); wp = [w.tolist()]
    else:
        idx = range(0, len(path), 2)
        two = [(path[i], path[j]) for i in idx for j in idx if j > i and ok(s0, path[i]) and ok(path[i], path[j]) and ok(path[j], g)]
        if two:
            a, b = min(two, key=lambda t: np.linalg.norm(t[0] - s0) + np.linalg.norm(t[1] - t[0]) + np.linalg.norm(g - t[1])); n = 2; wz = float(max(a[2], b[2])); wp = [a.tolist(), b.tolist()]
    res.append((tid, n, legs, wz if n else None, float(s0[2]), float(g[2]), wp))
    print(tid.split('__rl_train__')[1][:44], '中间点数', n, '两段长度 %s' % (None if not legs else '%.1f / %.1f m' % legs), '中间点高度 %s（起 %.1f 终 %.1f）' % (None if not n else '%.1f' % wz, s0[2], g[2]), flush=True)
c = [r[1] for r in res]
print('1 个中间点即可：%d；需要 2 个：%d；2 个也不行：%d（共 %d 个任务）' % (c.count(1), c.count(2), c.count(None), len(c)))
json.dump({r[0]: r[6] for r in res if r[6]}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'subgoals_dev420_blocked.json'), 'w'), indent=1)
