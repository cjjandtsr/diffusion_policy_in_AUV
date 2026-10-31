"""Plot executed trajectories of collision episodes in obstacle-free-corridor tasks.
Picks with random.Random(20261031): 2 dev P0, 2 test12 P0, 2 test12 P6.
Usage: python free_corridor_plot.py <main_checkout_root> <out.png>"""
import glob, json, math, random, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plots, stratify as S

def trajectory(directory, tid, rep):
    for f in sorted(glob.glob(directory + '/traces/*.jsonl')):
        with open(f) as h:
            st = json.loads(h.readline())
        if st['task_id'] == tid and int(st['repeat']) == rep:
            pts = []
            for line in open(f):
                x = json.loads(line)
                if x.get('type') == 'transition':
                    pts += x['executed_path'] if not pts else x['executed_path'][1:]
            last = x
    return pts

def main(root, out):
    rows = json.load(open('free_corridor_collisions.json'))
    geo = {t['task_id']: t for t in json.load(open('task_geometry.json'))}
    met = {m['task_id']: m for m in json.load(open('task_metrics.json'))}
    rng = random.Random(20261031); src = S.sources(root); picks = []
    for split, pol in (('dev420', 'P0'), ('test12', 'P0'), ('test12', 'P6')):
        C = [r for r in rows if r['split'] == split and r['policy'] == pol and r['outcome'] == 'collision']
        picks += [(split, pol, r) for r in rng.sample(C, 2)]
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 3.3 * len(picks)))
    for (split, pol, r), (a, b) in zip(picks, axes):
        t = geo[r['task_id']]; plots.draw(a, b, t, met[r['task_id']])
        tr = trajectory(src[split][pol], r['task_id'], r['repeat'])
        s, g = t['start'], t['goal']; L = math.hypot(g[0]-s[0], g[1]-s[1]); u = ((g[0]-s[0])/L, (g[1]-s[1])/L)
        a.plot([p[0] for p in tr], [p[1] for p in tr], 'm-', lw=1.5); a.plot(r['point'][0], r['point'][1], 'kx', ms=10, mew=2)
        b.plot([(p[0]-s[0])*u[0]+(p[1]-s[1])*u[1] for p in tr], [p[2] for p in tr], 'm-', lw=1.5)
        b.set_xlim(-2, max(L, max((p[0]-s[0])*u[0]+(p[1]-s[1])*u[1] for p in tr)) + 2)
        b.set_title('%s %s rep%d: magenta = executed trajectory, x = collision' % (split, pol, r['repeat']), fontsize=8)
        a.relim(); a.autoscale_view()
    fig.tight_layout(); fig.savefig(out, dpi=80)

if __name__ == '__main__':
    main(*sys.argv[1:3])
