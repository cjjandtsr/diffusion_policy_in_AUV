import json, math, collections, statistics as st
d = json.load(open('/projects/sandbox/scratch/eps.json'))
POL = ['bc', 'u32', 'u128', 'u256']
def q(a, f): a = sorted(a); return a[int(f * (len(a) - 1))]
print('## vertical behaviour: required dz = goal_z - start_z; achieved closest-approach dz; vertical travel')
for p in POL:
    res = collections.defaultdict(list)
    for k, e in d.items():
        if not k.startswith(p + '|'): continue
        s, g = e['taskd']['start_pos'], e['taskd']['goal_pos']
        pts = [pp for seg in e['path'] for pp in seg]
        i = min(range(len(pts)), key=lambda j: math.dist(pts[j], g)); pc = pts[i]
        o = 'R' if e['done'] == 'reached' else 'T' if e['done'] == 'time_limit' else 'C'
        md = math.dist(pc, g)
        cls = o if o != 'T' else ('T<1m' if md < 1 else 'T<2m' if md < 2 else 'Tfar')
        req = g[2] - s[2]
        zs = [pp[2] for pp in pts]
        res[cls].append(dict(req=req, dz=pc[2] - g[2], dxy=math.hypot(pc[0] - g[0], pc[1] - g[1]), zr=max(zs) - min(zs),
                             achieved=(pc[2] - s[2]) / req if abs(req) > .3 else None, gz=g[2], sz=s[2]))
    for cls in ['R', 'T<1m', 'T<2m', 'Tfar', 'C']:
        v = res[cls]
        if not v: continue
        ach = [x['achieved'] for x in v if x['achieved'] is not None]
        print(p, cls.ljust(5), 'n=%3d | goal z %.2f start z %.2f | req dz med %+.2f |req|>0.5: %3d | closest dz med %+.2f (|dz|>0.4: %d) dxy med %.2f | z range med %.2f | frac of required dz achieved med %.2f' % (
            len(v), st.mean(x['gz'] for x in v), st.mean(x['sz'] for x in v), q([x['req'] for x in v], .5), sum(abs(x['req']) > .5 for x in v),
            q([x['dz'] for x in v], .5), sum(abs(x['dz']) > .4 for x in v), q([x['dxy'] for x in v], .5), q([x['zr'] for x in v], .5), q(ach, .5) if ach else float('nan')))
print('\n## arrival rate by |required dz| bin (all 84 tasks x 5)')
for p in POL:
    b = collections.defaultdict(lambda: [0, 0])
    for k, e in d.items():
        if not k.startswith(p + '|'): continue
        s, g = e['taskd']['start_pos'], e['taskd']['goal_pos']; r = abs(g[2] - s[2])
        key = '<0.3' if r < .3 else '0.3-0.7' if r < .7 else '0.7-1.2' if r < 1.2 else '>=1.2'
        b[key][0] += e['done'] == 'reached'; b[key][1] += 1
    print(p, ' '.join('%s:%d/%d' % (k, *b[k]) for k in ['<0.3', '0.3-0.7', '0.7-1.2', '>=1.2']))
