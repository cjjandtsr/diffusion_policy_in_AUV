import json, math, collections, statistics as st
d = json.load(open('/projects/sandbox/scratch/eps.json'))
E = {tuple(k.split('|')): v for k, v in json.load(open('/projects/sandbox/scratch/E.json')).items()}
tasks = sorted({k[1] for k in E}); POL = ['bc', 'u32', 'u128', 'u256']
bcrate = {t: sum(E[('bc', t, r)]['o'] == 'R' for r in '12345') for t in tasks}
print('## gains/losses vs BC (paired task/rep), by BC difficulty and line clearance')
def strat(t):
    l = E[('bc', t, '1')]['line']; return ('BCfail' if bcrate[t] == 0 else 'BCok' if bcrate[t] == 5 else 'BCmix') + ('|line<0.5' if l < .5 else '|line0.5-1.5' if l < 1.5 else '|line>=1.5')
for p in POL[1:]:
    c = collections.Counter()
    for t in tasks:
        for r in '12345':
            a, b = E[('bc', t, r)]['o'], E[(p, t, r)]['o']
            if a != b: c[(strat(t), a + '>' + b)] += 1
    print(p, sorted(c.items()))
print('\n## near-goal timeouts (min dist<1m): closest-point decomposition, time spent near goal')
for p in POL:
    rows = []
    for k, e in d.items():
        q, t, r = k.split('|')
        if q != p or e['done'] != 'time_limit': continue
        g = e['taskd']['goal_pos']; pts = [pp for seg in e['path'] for pp in seg]
        i = min(range(len(pts)), key=lambda j: math.dist(pts[j], g)); pc = pts[i]
        dd = math.dist(pc, g)
        if dd >= 1: continue
        dxy = math.hypot(pc[0] - g[0], pc[1] - g[1]); dz = pc[2] - g[2]
        near = sum(1 for pos in e['dec_pos'] if math.dist(pos, g) < 2)
        firstnear = next(j for j, pos in enumerate(e['dec_pos'] + [pts[-1]]) if math.dist(pos, g) < 2) if near else None
        final = math.dist(pts[-1], g)
        # yaw relative to goal at decision of closest approach
        dec = min(i // 5, len(e['yaw']) - 1)
        pos = e['dec_pos'][dec]; bear = math.atan2(g[1] - pos[1], g[0] - pos[0]) - e['yaw'][dec]
        bear = abs(math.degrees(math.atan2(math.sin(bear), math.cos(bear))))
        rows.append(dict(t=t, dd=dd, dxy=dxy, dz=dz, near=near, final=final, bear=bear, n=len(e['path'])))
    if not rows: continue
    print(p, 'n=%d tasks=%d | closest %.2f  |dxy| %.2f  dz %+.2f (|dz|>|dxy|: %d) | decisions within 2m %.1f | final dist %.1f | bearing at closest %.0f deg' % (
        len(rows), len({x['t'] for x in rows}), st.mean(x['dd'] for x in rows), st.mean(x['dxy'] for x in rows), st.mean(x['dz'] for x in rows),
        sum(abs(x['dz']) > x['dxy'] for x in rows), st.mean(x['near'] for x in rows), st.mean(x['final'] for x in rows), st.mean(x['bear'] for x in rows)))
    print('   closest-dist hist', collections.Counter(round(x['dd'], 1) for x in rows).most_common(6),
          ' final>3m:', sum(x['final'] > 3 for x in rows), ' near-goal decisions>=10:', sum(x['near'] >= 10 for x in rows))
    print('   top tasks', collections.Counter(x['t'].split('__')[3][:20] + x['t'][-5:] for x in rows).most_common(6))
