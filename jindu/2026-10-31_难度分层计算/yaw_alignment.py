"""Heading (body yaw) vs horizontal travel direction, per decision, all episodes.

The policy observes a forward depth camera (synth_pointcloud.py: hfov 60 deg, far 10 m,
body frame), so travel more than 30 deg off the yaw axis is outside the camera's view.
misalign = |wrap(mean(yaw_pre, yaw_post) - heading of the decision's executed horizontal displacement)|,
for decisions that move > 0.1 m horizontally (1 decision = 1.2 s).
Usage: python yaw_alignment.py <main_checkout_root> <out.json>
"""
import glob, json, math, sys
import numpy as np
import stratify as S

def run(d):
    mis, yawrate, seen = [], [], set()
    for f in sorted(glob.glob(d + '/traces/*.jsonl'), reverse=True):  # later attempt first
        st = None; rows = []
        for line in open(f):
            x = json.loads(line)
            if x.get('type') == 'episode_start': st = x
            elif x.get('type') == 'transition': rows.append(x)
        key = (st['task_id'], st['repeat'])
        if key in seen or not rows or rows[-1]['done_reason'] == 'none': continue
        seen.add(key)
        for x in rows:
            a, b = x['executed_path'][0], x['executed_path'][-1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if math.hypot(dx, dy) > 0.1:
                y0, y1 = x['pre_state']['yaw'], x['post_state']['yaw']
                ym = y0 + ((y1 - y0 + math.pi) % (2 * math.pi) - math.pi) / 2
                mis.append(abs(math.degrees((ym - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi)))
            yawrate.append(abs(math.degrees((x['post_state']['yaw'] - x['pre_state']['yaw'] + math.pi) % (2 * math.pi) - math.pi)))
    m = np.array(mis)
    return dict(episodes=len(seen), decisions=len(m), median_deg=float(np.median(m)), frac_outside_30=float(np.mean(m > 30)),
                frac_outside_90=float(np.mean(m > 90)), yaw_change_per_decision_median_deg=float(np.median(yawrate)))

def main(root, out):
    src = S.sources(root)
    a1 = '/projects/sandbox/diffusion_policy_in_AUV/jindu/routea_common_a1_v1_run01/evaluation/full/'
    src['dev420']['P2@u32'] = a1 + 'rl_391a6656a0f7_u32'
    res = {split: {p: run(d) for p, d in pols.items()} for split, pols in src.items()}
    json.dump(res, open(out, 'w'), indent=1)
    for split, r in res.items():
        for p, v in r.items():
            print(split, p.ljust(7), 'median %.1f deg | outside 30 deg %.2f | outside 90 deg %.2f | yaw change/decision %.1f deg' % (
                v['median_deg'], v['frac_outside_30'], v['frac_outside_90'], v['yaw_change_per_decision_median_deg']))

if __name__ == '__main__':
    main(*sys.argv[1:3])
