import json, glob, math, collections, statistics as st
B = '/projects/sandbox/diffusion_policy_in_AUV/jindu/2026-10-25_P1_P7信息核查/q2_q3_q4/q2/'
D = {}
for f in glob.glob(B + '*/*/*.json'):
    d = json.load(open(f))
    if 'mode' not in d: continue
    D[(d['mode'], f.split('/')[-2], d['state_id'])] = d
pol = ['bc', 'rl_391a6656a0f7_u128', 'rl_391a6656a0f7_u256']
sh = dict(zip(pol, ['bc', 'u128', 'u256']))
states = sorted({k[2] for k in D}); grp = {k[2]: D[k]['group'] for k in D}

def vec(s, a, b):
    A = s['full_action_16x11']
    return [sum(A[i][c] for i in range(a, b)) for c in range(3)]
def yaw(s, a, b):
    return sum(s['full_action_16x11'][i][9] for i in range(a, b))
def norm(v): return math.sqrt(sum(x * x for x in v))
def sub(a, b): return [x - y for x, y in zip(a, b)]
def mean(vs): return [sum(v[i] for v in vs) / len(vs) for i in range(len(vs[0]))]
def rms(vs):
    m = mean(vs); return math.sqrt(sum(norm(sub(v, m)) ** 2 for v in vs) / len(vs))
def angdiff(a, b):
    d = a - b; return abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))

R = collections.defaultdict(list)
for m in ['deterministic', 'random']:
    for p in pol:
        for s in states:
            S = D[(m, p, s)]['samples']
            h4 = [vec(x, 2, 6) for x in S]; h8 = [vec(x, 2, 10) for x in S]
            mh = mean(h4); ma = math.atan2(mh[1], mh[0])
            far = sum(angdiff(math.atan2(v[1], v[0]), ma) > 30 for v in h4)
            mp = max(norm(sub(a, b)) for a in h8 for b in h8)
            ys = st.pstdev([yaw(x, 2, 6) for x in S])
            R[(m, p)].append(dict(s=s, g=grp[s], rms4=rms(h4), rms8=rms(h8), mag=sum(map(norm, h4)) / 20,
                                  far=far, maxp=mp, ysd=math.degrees(ys)))
f = lambda r, k: sum(x[k] for x in r) / len(r)
print('mode policy | rms4 rms8 |H4| yawSD(deg) maxpairH8 draws>30deg/360')
for (m, p), r in R.items():
    print(m[:4], sh[p], '| %.4f %.4f %.3f %.2f %.3f %d' % (f(r, 'rms4'), f(r, 'rms8'), f(r, 'mag'), f(r, 'ysd'), f(r, 'maxp'), sum(x['far'] for x in r)))
print('\nby group: rand rms8 / rand maxpairH8 / det rms8 / |H4|')
for p in pol:
    for g in sorted(set(grp.values())):
        r = [x for x in R[('random', p)] if x['g'] == g]; d = [x for x in R[('deterministic', p)] if x['g'] == g]
        print(sh[p], g, '%.4f %.3f %.4f %.3f' % (f(r, 'rms8'), f(r, 'maxp'), f(d, 'rms8'), f(r, 'mag')))
print('\npaired same-draw H4 diff')
for p in pol:
    v = [st.mean(norm(sub(vec(a, 2, 6), vec(b, 2, 6))) for a, b in zip(D[('deterministic', p, s)]['samples'], D[('random', p, s)]['samples'])) for s in states]
    print(sh[p], 'det vs rand %.4f' % st.mean(v))
for a, b in [(0, 1), (1, 2), (0, 2)]:
    for m in ['deterministic', 'random']:
        v = [st.mean(norm(sub(vec(x, 2, 6), vec(y, 2, 6))) for x, y in zip(D[(m, pol[a], s)]['samples'], D[(m, pol[b], s)]['samples'])) for s in states]
        print(sh[pol[a]], 'vs', sh[pol[b]], m[:4], '%.4f' % st.mean(v))
print('\nper-state random rms8 / maxpairH8 (bc u128 u256)')
for i, s in enumerate(states):
    print(grp[s][:16].ljust(16), s[:6], '  '.join('%.3f/%.2f' % (R[('random', p)][i]['rms8'], R[('random', p)][i]['maxp']) for p in pol))
print('\nBC det draws>30deg by state:',[(grp[x['s']][:8],x['s'][:6],x['far']) for x in R[('deterministic','bc')] if x['far']])
print('u256 rand far:',[(grp[x['s']][:8],x['s'][:6],x['far']) for x in R[('random','rl_391a6656a0f7_u256')] if x['far']])
