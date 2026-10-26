import json, math, collections, statistics as st
d = json.load(open('/projects/sandbox/scratch/eps.json'))
POL = ['bc', 'u32', 'u128', 'u256']

def clear_pt(p, o):
    dx, dy, dz = p[0] - o['x'], p[1] - o['y'], p[2] - o['z']
    if o['shape'] == 'cylinder':
        xy = max(math.hypot(dx, dy) - o['radius'], 0.); hz = o['height'] / 2 + .1
    else:
        c, s = math.cos(o['yaw']), math.sin(o['yaw'])
        x, y = c * dx + s * dy, -s * dx + c * dy
        xy = math.hypot(max(abs(x) - o['size_x'] / 2, 0.), max(abs(y) - o['size_y'] / 2, 0.)); hz = o['size_z'] / 2 + .1
    return math.hypot(xy, max(abs(dz) - hz, 0.))

def seg_clear(a, b, geom, n=8):
    best = 1e9
    for i in range(n + 1):
        t = i / n; p = [a[k] + t * (b[k] - a[k]) for k in range(3)]
        for o in geom: best = min(best, clear_pt(p, o))
    return best

def outc(e):
    r = e['done']; return 'R' if r == 'reached' else 'C' if 'coll' in r or 'crash' in r else 'T' if r == 'time_limit' else r

E = {}
line_cache = {}
for k, e in d.items():
    pol, task, rep = k.split('|')
    s, g = e['taskd']['start_pos'], e['taskd']['goal_pos']
    if task not in line_cache:
        line_cache[task] = seg_clear(s, g, e['geom'], 200)
    pts = [p for seg in e['path'] for p in seg]
    minc = min(seg_clear(a, b, e['geom'], 4) for seg in e['path'] for a, b in zip(seg, seg[1:]))
    gd = [math.dist(p, g) for p in pts]
    D0 = math.dist(s, g)
    steps = [math.dist(a, b) for a, b in zip(e['dec_pos'], e['dec_pos'][1:])]
    # first entry into 2 m sphere: radial speed (m per decision) over the decision containing entry
    entry = None
    for i, seg in enumerate(e['path']):
        if any(math.dist(p, g) < 2. for p in seg):
            a = seg[0]; b = seg[-1]
            entry = dict(i=i, speed=math.dist(a, b), radial=math.dist(a, g) - math.dist(b, g)); break
    E[(pol, task, rep)] = dict(o=outc(e), role=e['role'], line=line_cache[task], D0=D0, minc=minc,
                              mind=min(gd), step=st.mean(steps) if steps else 0, n=len(e['path']), entry=entry,
                              world=task.split('__')[3])
tasks = sorted({k[1] for k in E})
bcrate = {t: sum(E[('bc', t, r)]['o'] == 'R' for r in '12345') for t in tasks}
def strata_line(x): return '<0.5' if x < .5 else '0.5-1.5' if x < 1.5 else '1.5-3' if x < 3 else '>=3'
def table(name, fn):
    print('\n## by', name, '(R/C/T counts per policy)')
    groups = sorted({fn(t) for t in tasks}, key=str)
    for gname in groups:
        ts = [t for t in tasks if fn(t) == gname]
        cells = []
        for p in POL:
            c = collections.Counter(E[(p, t, r)]['o'] for t in ts for r in '12345')
            cells.append('%s %d/%d/%d' % (p, c['R'], c['C'], c['T']))
        print(str(gname).ljust(28), 'n=%3d' % (5 * len(ts)), ' | '.join(cells))
table('role', lambda t: E[('bc', t, '1')]['role'])
table('straight-line clearance (m)', lambda t: strata_line(E[('bc', t, '1')]['line']))
table('BC success count /5 (difficulty proxy)', lambda t: bcrate[t])

print('\n## mean min path clearance (m), non-collision episodes, paired with BC same task/rep, by line-clearance stratum')
for sname in ['<0.5', '0.5-1.5', '1.5-3', '>=3']:
    ts = [t for t in tasks if strata_line(E[('bc', t, '1')]['line']) == sname]
    row = []
    for p in POL[1:]:
        diffs = [E[(p, t, r)]['minc'] - E[('bc', t, r)]['minc'] for t in ts for r in '12345' if E[(p, t, r)]['o'] != 'C' and E[('bc', t, r)]['o'] != 'C']
        row.append('%s %+.2f (n=%d)' % (p, st.mean(diffs), len(diffs)) if diffs else p + ' -')
    bcm = [E[('bc', t, r)]['minc'] for t in ts for r in '12345' if E[('bc', t, r)]['o'] != 'C']
    print(sname.ljust(8), 'bc mean %.2f |' % st.mean(bcm), ' | '.join(row))

print('\n## collision tasks of u256 not collided by bc: min clearance per policy (mean over 5 reps) and outcomes')
ct = sorted({t for t in tasks if sum(E[('u256', t, r)]['o'] == 'C' for r in '12345') >= 2})
for t in ct:
    print(t.split('__')[3][:26].ljust(26), t[-5:], 'line %.2f' % E[('bc', t, '1')]['line'],
          ' '.join('%s:%s %.2f' % (p, ''.join(E[(p, t, r)]['o'] for r in '12345'), st.mean(E[(p, t, r)]['minc'] for r in '12345')) for p in POL))

print('\n## step length (m/decision) by policy, all episodes, and BC-success>=4 tasks')
for p in POL:
    print(p, '%.2f' % st.mean(x['step'] for (q, t, r), x in E.items() if q == p),
          'easy %.2f' % st.mean(x['step'] for (q, t, r), x in E.items() if q == p and bcrate[t] >= 4))

print('\n## near goal: episodes that entered 2 m sphere; outcome and entry speed / radial approach (m per 1.2 s)')
for p in POL:
    ent = [x for (q, t, r), x in E.items() if q == p and x['entry']]
    for o in 'RTC':
        v = [x for x in ent if x['o'] == o]
        if v: print(p, o, len(v), 'speed %.2f radial %.2f' % (st.mean(x['entry']['speed'] for x in v), st.mean(x['entry']['radial'] for x in v)))
    tm = [x for (q, t, r), x in E.items() if q == p and x['o'] == 'T']
    print(p, 'timeouts', len(tm), 'mind<1m', sum(x['mind'] < 1 for x in tm), 'mind<2m', sum(x['mind'] < 2 for x in tm))

print('\n## timeouts that came within 1 m: speed at entry quantiles vs reached')
for p in POL:
    r = sorted(x['entry']['speed'] for (q, t, rr), x in E.items() if q == p and x['entry'] and x['o'] == 'R')
    tt = sorted(x['entry']['speed'] for (q, t, rr), x in E.items() if q == p and x['entry'] and x['o'] == 'T' and x['mind'] < 1)
    q = lambda a, f: a[int(f * (len(a) - 1))] if a else float('nan')
    print(p, 'reached med %.2f p75 %.2f | near-timeouts n=%d med %.2f p75 %.2f' % (q(r, .5), q(r, .75), len(tt), q(tt, .5), q(tt, .75)))
json.dump({'|'.join(k): v for k, v in E.items()}, open('/projects/sandbox/scratch/E.json', 'w'))
