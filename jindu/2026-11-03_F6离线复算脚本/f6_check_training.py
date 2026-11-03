"""F6 §5: gates (sum identities, continuity, straight-line reachability inside R) and diagnostics on training rollouts.
Usage: python3 -I f6_check_training.py <REPO_ROOT> <REPO_ROOT>/jindu/routea_f5_run01/results/prepared/training/P9 [R=2.0]"""
import json, sys, os, glob, statistics as st, collections
repo, mdir = sys.argv[1], sys.argv[2]
R = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
K, G = 10.0, 0.99
sys.path.insert(0, os.path.join(repo, 'Improved-3D-Diffusion-Policy'))
from diffusion_policy_3d.env.collision_geom import segment_obstacle_collision, ObstacleSpec
wdir = os.path.join(repo, 'PX4-Autopilot-Tools/sitl_gazebo/worlds/train')
geo = {}
def obstacles(w):
    if w not in geo:
        geo[w] = [ObstacleSpec.from_dict(o) for o in json.load(open(os.path.join(wdir, w + '.world.obstacles.json')))['obstacles']]
    return geo[w]
phi = lambda d: K * (1 - min(d, R) / R)

eps = collections.OrderedDict()
for f in sorted(glob.glob(os.path.join(mdir, 'metrics_update_*.json'))):
    for r in json.load(open(f))['rollout_audit']['transition_records']:
        eps.setdefault(r['episode_id'], []).append(r)

gap = err_u = err_d = 0.0; los_ok = los_bad = 0; D = collections.defaultdict(list); space = []; d0_in = 0; n_ep = 0
for T in eps.values():
    T.sort(key=lambda r: (r['update_index'], int(r['transition_id'].split(':t')[1])))
    last = T[-1]
    if not (last['terminated'] or last['truncated']):
        continue                                            # partial segment at a collection boundary
    n_ep += 1
    for a, b in zip(T[:-1], T[1:]):
        gap = max(gap, abs(a['goal_distance_after_3d'] - b['goal_distance_before_3d']))
    dist = [T[0]['goal_distance_before_3d']] + [r['goal_distance_after_3d'] for r in T]   # d_0 .. d_T
    P = [phi(d) for d in dist]; n = len(T)
    F = [G * P[t + 1] - P[t] for t in range(n)]
    err_u = max(err_u, abs(sum(F) - (G * P[n] - P[0] - (1 - G) * sum(P[1:n]))))
    err_d = max(err_d, abs(sum(G ** t * F[t] for t in range(n)) - (G ** n * P[n] - P[0])))
    d0_in += P[0] > 0
    space.append(sum(r['reward_terms'].get('space', 0.) for r in T))
    # straight-line reachability from the first decision endpoint inside R (no earlier physical failure)
    g = last['goal_xyz']
    q = T[0]['position_before_xyz'] if dist[0] <= R else next(
        (r['position_after_xyz'] for r in T if r['goal_distance_after_3d'] <= R
         and r['terminal_facts']['done_reason'] in ('none', 'reached', 'time_limit')), None)
    if q is not None:
        clear = min(q[2], g[2]) >= 0.2 and segment_obstacle_collision(q, g, obstacles(last['world_id']), 0.1) is None
        los_ok += clear; los_bad += not clear
    dr = last['terminal_facts']['done_reason']; k = min(range(len(dist)), key=dist.__getitem__)
    oc = ('reached' if dr == 'reached' else
          ('collision_endR' if dist[-1] <= R else 'collision_far') if last['terminated'] else
          ('TL_nearmiss' if min(dist) < 1.0 else 'TL_other'))
    D[oc].append(dict(sumF=sum(F), end=G * P[n] - P[0], leak=-(1 - G) * sum(P[1:n]),
                      pre=sum(F[:k]), post=sum(F[k:]), v4=sum(r['reward'] for r in T) + sum(F)))

print('complete episodes %d (of %d)' % (n_ep, len(eps)))
print('GATE identities: max err undiscounted %.2e, discounted %.2e; continuity max gap %.2e' % (err_u, err_d, gap))
print('GATE reachability R=%.1f: %d/%d clear (%.1f%%)' % (R, los_ok, los_ok + los_bad, 100 * los_ok / max(1, los_ok + los_bad)))
print('DIAG D0<=R: %d; space term per episode min %.4f median %.4f' % (d0_in, min(space), st.median(space)))
for oc in ('reached', 'TL_nearmiss', 'TL_other', 'collision_endR', 'collision_far'):
    v = D[oc]
    if not v: continue
    m = lambda key: st.median(x[key] for x in v); pct = lambda c: 100 * sum(map(c, v)) / len(v)
    print('DIAG %-14s n=%4d sumF %+.2f = end %+.2f + leak %+.2f | sumF>0 %.1f%% pre>0 %.1f%% post<0 %.1f%% | v4 return %+.2f' % (
        oc, len(v), m('sumF'), m('end'), m('leak'), pct(lambda x: x['sumF'] > 0), pct(lambda x: x['pre'] > 0),
        pct(lambda x: x['post'] < 0), m('v4')))
