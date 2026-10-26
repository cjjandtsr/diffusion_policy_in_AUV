import csv, collections, statistics as st, json
B = '/projects/sandbox/diffusion_policy_in_AUV/jindu/2026-10-25_P1_P7信息核查/'
E = [r for r in csv.DictReader(open(B + 'training_episode_outcomes.csv',encoding='utf-8-sig')) if r['collection_boundary']=='False']
print('roles', collections.Counter(r['role'] for r in E), 'outcomes', collections.Counter(r['physical_outcome'] for r in E))
for arm in ('b2560', 'b1280'):
    print('\n==', arm, 'block | per role success% (n) | collision% | reached: mean decisions per 10m of D0 | distinct succeeding (world,role) in dense')
    rs = [r for r in E if arm in r['arm']]
    for b in range(8):
        blk = [r for r in rs if b * 32 < int(r['update']) <= (b + 1) * 32]
        roles = sorted({r['role'] for r in blk}); s = []
        for ro in roles:
            x = [r for r in blk if r['role'] == ro]
            s.append('%s %.0f%%/%.0f%%(%d)' % (ro[:6], 100 * sum(r['physical_outcome'] == 'reached' for r in x) / len(x),
                     100 * sum('coll' in r['physical_outcome'] or 'ground' in r['physical_outcome'] for r in x) / len(x), len(x)))
        R = [r for r in blk if r['physical_outcome'] == 'reached']
        pace = st.mean(int(r['recorded_transitions']) / float(r['start_distance']) * 10 for r in R) if R else 0
        T = [r for r in blk if r['physical_outcome'] == 'none']
        tend = st.mean(float(r['end_distance']) for r in T) if T else 0
        dense = {r['world_id'] for r in R if r['role'] == 'dense_recovery'}
        print('u%3d-%3d' % (b * 32 + 1, b * 32 + 32), ' | '.join(s), '| pace %.2f | timeout end_dist %.1f | dense succ worlds %d' % (pace, tend, len(dense)))
