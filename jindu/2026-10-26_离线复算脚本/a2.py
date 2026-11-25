import json, math, collections, statistics as st
d = json.load(open('/projects/sandbox/scratch/eps.json')); d.update(json.load(open('/projects/sandbox/scratch/eps_a2.json')))
POL = ['bc', 'u128', 'u256', 'C', 'T']
def oc(e): r = e['done']; return 'R' if r == 'reached' else 'T' if r == 'time_limit' else 'C'
tasks = sorted({k.split('|')[1] for k in d})
def ep(p, t, r): return d['%s|%s|%s' % (p, t, r)]
# determinism check
for a, b in [('bc', 'bc2'), ('u128', 'u128b')]:
    print('repro', a, b, sum(oc(ep(a, t, r)) == oc(ep(b, t, r)) for t in tasks for r in '12345'), '/420')
def clear_pt(p, o):
    dx, dy, dz = p[0] - o['x'], p[1] - o['y'], p[2] - o['z']
    if o['shape'] == 'cylinder': xy = max(math.hypot(dx, dy) - o['radius'], 0.); hz = o['height'] / 2 + .1
    else:
        c, s = math.cos(o['yaw']), math.sin(o['yaw']); x, y = c * dx + s * dy, -s * dx + c * dy
        xy = math.hypot(max(abs(x) - o['size_x'] / 2, 0.), max(abs(y) - o['size_y'] / 2, 0.)); hz = o['size_z'] / 2 + .1
    return math.hypot(xy, max(abs(dz) - hz, 0.))
line = {}
for t in tasks:
    e = ep('bc', t, '1'); s, g = e['taskd']['start_pos'], e['taskd']['goal_pos']
    line[t] = min(clear_pt([s[k] + i / 200 * (g[k] - s[k]) for k in range(3)], o) for i in range(201) for o in e['geom'])
gz = {t: ep('bc', t, '1')['taskd']['goal_pos'][2] for t in tasks}
bcr = {t: sum(oc(ep('bc', t, r)) == 'R' for r in '12345') for t in tasks}
def row(name, ts):
    cells = []
    for p in POL:
        c = collections.Counter(oc(ep(p, t, r)) for t in ts for r in '12345'); cells.append('%s %3d/%3d/%3d' % (p, c['R'], c['C'], c['T']))
    print(name.ljust(34), 'n=%3d' % (5 * len(ts)), ' | '.join(cells))
print('\nR/C/T')
row('ALL', tasks)
row('goal>=3, line clear>=0.5', [t for t in tasks if gz[t] >= 3 and line[t] >= .5])
row('goal>=3, line blocked<0.5', [t for t in tasks if gz[t] >= 3 and line[t] < .5])
row('goal<3, line clear', [t for t in tasks if gz[t] < 3 and line[t] >= .5])
row('goal<3, line blocked', [t for t in tasks if gz[t] < 3 and line[t] < .5])
row('goal>=3.5', [t for t in tasks if gz[t] >= 3.5])
row('BC 5/5 tasks', [t for t in tasks if bcr[t] == 5])
row('BC 0/5 tasks', [t for t in tasks if bcr[t] == 0])
narrow = [t for t in tasks if bcr[t] >= 3 and 0.3 <= line[t] < 1.5]
row('narrow-margin BC-ok (line .3-1.5)', narrow)
dense_blk = [t for t in tasks if ep('bc', t, '1')['role'] == 'dense_recovery' and line[t] < .1]
row('dense & line<0.1', dense_blk)
print('\npaired vs u128: kept / lost / gained (successes)')
for p in ['u256', 'C', 'T']:
    k = l = g = 0
    for t in tasks:
        for r in '12345':
            a, b = oc(ep('u128', t, r)) == 'R', oc(ep(p, t, r)) == 'R'; k += a and b; l += a and not b; g += b and not a
    kb = lb = 0
    for t in tasks:
        for r in '12345':
            a, b = oc(ep('bc', t, r)) == 'R', oc(ep(p, t, r)) == 'R'; kb += a and b; lb += a and not b
    print(p, 'vs u128 kept %d lost %d gained %d | vs BC kept %d lost %d' % (k, l, g, kb, lb))
print('\nnear-goal below-goal timeouts (min dist<1 m, dz<-0.3 at closest)')
for p in POL:
    n = n2 = 0
    for t in tasks:
        for r in '12345':
            e = ep(p, t, r)
            if e['done'] != 'time_limit': continue
            g = e['taskd']['goal_pos']; pc = min((q for s in e['path'] for q in s), key=lambda q: math.dist(q, g))
            if math.dist(pc, g) < 1: n += 1; n2 += pc[2] - g[2] < -.3
    print(p, 'near<1m timeouts', n, 'below-goal', n2)
print('\nper-task outcomes on narrow-margin tasks')
for t in narrow:
    print(t.split('__')[3][:24].ljust(24), t[-5:], 'line %.2f' % line[t], ' '.join(p + ':' + ''.join(oc(ep(p, t, r)) for r in '12345') for p in POL))
print('\nper-task (goal>=3, clear) reached counts /5')
for t in [t for t in tasks if gz[t] >= 3 and line[t] >= .5]:
    print(t.split('__')[3][:24].ljust(24), t[-5:], 'gz %.2f' % gz[t], ' '.join('%s:%d' % (p, sum(oc(ep(p, t, r)) == 'R' for r in '12345')) for p in POL))
