"""F6 §0: v2 return decomposition by outcome on dev420 #E2 traces (near-miss = time_limit with executed path within 1 m of goal).
Usage: python3 -I f6_dev_v2_returns.py <REPO_ROOT>/jindu/routea_f5_run01/results/eval_dev/full/P6/traces"""
import json, sys, os, ast, statistics as st
import numpy as np
d = sys.argv[1]; G = {}; post = []
for f in os.listdir(d):
    L = [json.loads(l) for l in open(os.path.join(d, f))]
    s, e = L[0], L[-1]
    task = s['task'] if isinstance(s['task'], dict) else ast.literal_eval(s['task'])
    g = np.asarray(task['goal_pos'], float)
    T = [t for t in L[1:] if t.get('type') == 'transition']
    terms = [t['reward_terms'] if isinstance(t['reward_terms'], dict) else ast.literal_eval(t['reward_terms']) for t in T]
    dmin = [min(np.linalg.norm(np.asarray(q) - g) for q in t['executed_path']) for t in T]
    k = int(np.argmin(dmin)); oc = e['done_reason']
    if oc == 'time_limit': oc = 'TL_nearmiss' if dmin[k] < 1.0 else 'TL_other'
    G.setdefault(oc, []).append(({kk: sum(x.get(kk, 0) for x in terms) for kk in terms[0]}, sum(t['reward'] for t in T), len(T)))
    if oc == 'TL_nearmiss': post += [sum(x.values()) for x in terms[k + 1:]]
for oc, R in sorted(G.items()):
    print('%-12s n=%3d return %7.2f | ' % (oc, len(R), st.median(r[1] for r in R)) +
          ' '.join('%s %.2f' % (kk, st.median(r[0][kk] for r in R)) for kk in R[0][0]) + ' | decisions %d' % st.median(r[2] for r in R))
if post: print('near-miss after closest approach: %d decisions, mean reward %.3f' % (len(post), np.mean(post)))
