import csv, math, collections, statistics as st
B = '/projects/sandbox/diffusion_policy_in_AUV/jindu/2026-10-25_P1_P7信息核查/q2_q3_q4/'
T = collections.defaultdict(list)
for r in csv.DictReader(open(B + 'q4_per_decision_300.csv')):
    T[(r['policy'], r['task_id'], r['repeat'])].append(r)
sh = lambda p: p.split('_')[-1]
def f(r, k): return float(r[k])
def bearing(r):
    dx, dy = f(r, 'goal_x') - f(r, 'pos_x'), f(r, 'goal_y') - f(r, 'pos_y')
    d = math.atan2(dy, dx) - f(r, 'yaw'); return abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))
def dist(r): return math.dist([f(r, c) for c in ('pos_x', 'pos_y', 'pos_z')], [f(r, c) for c in ('goal_x', 'goal_y', 'goal_z')])
def lateral(r, s, g):
    p = [f(r, c) for c in ('pos_x', 'pos_y', 'pos_z')]
    v = [b - a for a, b in zip(s, g)]; w = [b - a for a, b in zip(s, p)]
    L = math.sqrt(sum(x * x for x in v)); t = sum(a * b for a, b in zip(v, w)) / L ** 2
    return math.dist(p, [a + t * b for a, b in zip(s, v)]), t
out = collections.defaultdict(collections.Counter); stat = collections.defaultdict(list)
for (p, task, rep), rows in T.items():
    rows.sort(key=lambda r: int(r['decision']))
    end = [r['done_reason'] for r in rows if r['done_reason'] != 'none']
    reason = end[-1] if end else rows[-1]['done_reason']
    reason = 'collision' if reason.startswith('coll') else reason
    out[sh(p)][reason] += 1
    s = [f(rows[0], c) for c in ('pos_x', 'pos_y', 'pos_z')]; g = [f(rows[0], c) for c in ('goal_x', 'goal_y', 'goal_z')]
    steps = [math.dist([f(a, c) for c in ('pos_x', 'pos_y', 'pos_z')], [f(b, c) for c in ('pos_x', 'pos_y', 'pos_z')]) for a, b in zip(rows, rows[1:])]
    lat = [lateral(r, s, g) for r in rows]
    stat[(sh(p), reason)].append(dict(
        n=len(rows), D0=math.dist(s, g), step=st.mean(steps) if steps else 0,
        maxlat=max(l for l, _ in lat), last_lat=lat[-1][0], last_t=lat[-1][1],
        lastdist=dist(rows[-1]), mindist=min(dist(r) for r in rows),
        bear_last=bearing(rows[-1]), bear_max=max(bearing(r) for r in rows),
        frac_bear90=sum(bearing(r) > 90 for r in rows) / len(rows), task=task, rep=rep,
        traj=[(round(dist(r), 1), round(bearing(r)), round(lateral(r, s, g)[0], 1)) for r in rows]))
for p in ('bc', 'u128', 'u256'): print(p, dict(out[p]))
keys = ['n', 'D0', 'step', 'maxlat', 'last_lat', 'last_t', 'lastdist', 'mindist', 'bear_last', 'bear_max', 'frac_bear90']
print('policy outcome count | ' + ' '.join(keys))
for (p, o), v in sorted(stat.items()):
    print(p, o, len(v), '| ' + ' '.join('%.2f' % st.mean(x[k] for x in v) for k in keys))
print('\nu256 collisions detail: task rep n D0 step maxlat lastdist mindist bear_last  | last 6 decisions (dist,bearing,lat)')
for p in ('u128', 'u256'):
    for x in sorted(stat[(p, 'collision')], key=lambda x: x['task']):
        print(p, x['task'].split('__')[1][:6], x['task'].split('__')[3][:22], x['task'][-5:], x['rep'], x['n'], '%.1f %.2f %.1f %.1f %.1f %d' % (x['D0'], x['step'], x['maxlat'], x['lastdist'], x['mindist'], x['bear_last']), x['traj'][-6:])
# same task-rep paired: which tasks u256 collides that bc reached
print('\npaired outcome transitions bc->u128->u256 by task (counts)')
pairs = collections.Counter()
for (p, task, rep) in T:
    if sh(p) != 'bc': continue
    o = []
    for q in ('bc', 'rl_391a6656a0f7_u128', 'rl_391a6656a0f7_u256'):
        rows = T[(q, task, rep)]; e = [r['done_reason'] for r in rows if r['done_reason'] != 'none']
        rr = e[-1] if e else 'x'; o.append('C' if rr.startswith('coll') else rr[0].upper())
    pairs['>'.join(o)] += 1
print(pairs.most_common())
print('\nper-task collisions u256 (task: n_coll/5):', collections.Counter(x['task'].split('__')[3] + x['task'][-5:] for x in stat[('u256', 'collision')]))
print('\nrepeat consistency per task (rest_clear): tasks with all 5 same outcome')
for q in ('bc','rl_391a6656a0f7_u128','rl_391a6656a0f7_u256'):
    by=collections.defaultdict(list)
    for (p,task,rep),rows in T.items():
        if p!=q: continue
        e=[r['done_reason'] for r in rows if r['done_reason']!='none']; by[task].append(e[-1][:4] if e else 'x')
    print(sh(q), sum(len(set(v))==1 for v in by.values()),'/',len(by))
for p in ('bc','u128','u256'):
    v=stat[(p,'time_limit')]
    print(p,'timeouts',len(v),'mindist<1.5:',sum(x['mindist']<1.5 for x in v),' frac_goal_behind>50% :',sum(x['frac_bear90']>.5 for x in v), ' mean step %.2f'%st.mean(x['step'] for x in v))
