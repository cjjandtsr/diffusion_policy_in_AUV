"""Offline behavior segmentation of existing committed Route A traces.

No policy execution, reward change or trajectory repair. Visibility absence is
a conservative bounding-sphere certificate, not a claim that all other views
contain obstacles. Physical geometry and the historical renderer are separate.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision


def runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1).tolist(),
                    np.flatnonzero(np.diff(padded) == -1).tolist()))


def first_pair(mask):
    """One-based first segment of two consecutive qualifying segments."""
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask[:-1] & mask[1:])
    return int(indices[0]+1) if len(indices) else None


def plan_features(path, goal, yaws, velocity, yaw):
    """Describe H4/H8 and the remaining generated suffix, without executing it."""
    path, goal = np.asarray(path), np.asarray(goal)
    distance = np.linalg.norm(path-goal, axis=1)
    progress = distance[:-1]-distance[1:]
    delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta[:, :2], axis=1)
    reference = np.asarray(velocity)[:2]
    reference_source = 'pre_velocity'
    if np.linalg.norm(reference)<.1:
        usable = np.flatnonzero(lengths>.01)
        reference = delta[usable[0], :2] if len(usable) else np.zeros(2)
        reference_source = 'first_moving_plan_segment'
    denom = lengths*np.linalg.norm(reference)
    cosine = np.divide(delta[:, :2]@reference, denom,
                       out=np.ones(len(delta)), where=denom>1e-12)
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    yaw_changes = np.degrees(np.abs(np.arctan2(np.sin(np.asarray(yaws)-yaw),
                                             np.cos(np.asarray(yaws)-yaw))))
    return dict(approach_onset=first_pair(progress>.01),
        approach_onset_sensitivity={str(t):first_pair(progress>t) for t in (.005, .02)},
        flight_turn30_onset=first_pair((angles>=30)&(lengths>.01)),
        flight_turn60_onset=first_pair((angles>=60)&(lengths>.01)),
        flight_direction_reference=reference_source,
        yaw_turn30_onset=first_pair(yaw_changes>=30),
        h4_progress=float(distance[0]-distance[4]),
        tail_progress=float(distance[4]-distance[8]),
        h8_progress=float(distance[0]-distance[8]),
        generated14_progress=float(distance[0]-distance[-1]),
        segment_progress=progress.tolist(),
        waypoint_distances=distance.tolist(),
        yaw_change_deg=yaw_changes.tolist())


def compare_replans(old_path, new_path, actual_post, goal):
    old, new, post, goal = map(np.asarray, (old_path, new_path, actual_post, goal))
    gap = post-old[4]
    difference = new[1:5]-old[5:9]
    a, b = old[8]-old[4], new[4]-new[0]
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return dict(overlap_rms_m=float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
        translated_overlap_rms_m=float(np.sqrt(np.mean(np.sum((difference-gap)**2, axis=1)))),
        execution_endpoint_error_m=float(np.linalg.norm(gap)),
        old_tail_progress=float(np.linalg.norm(old[4]-goal)-np.linalg.norm(old[8]-goal)),
        new_prefix_progress=float(np.linalg.norm(new[0]-goal)-np.linalg.norm(new[4]-goal)),
        net_displacement_cosine=float(np.dot(a,b)/denom) if denom>1e-10 else None)


def phase_label(clear, minimum_run=3):
    if clear[0]:
        return "initially_clear_always_clear" if all(clear) else "initially_clear_later_blocked"
    rs = runs(clear)
    if not rs:
        return "initially_blocked_never_clear"
    sustained = [(a, b) for a, b in rs if b-a >= minimum_run]
    if not sustained:
        return "initially_blocked_brief_clear_only"
    return ("initially_blocked_final_clear_run" if sustained[-1][1] == len(clear)
            else "initially_blocked_clear_then_reblocked")


class Scene:
    def __init__(self, geometry):
        self.obstacles = [ObstacleSpec.from_dict(o) for o in geometry]
        self.geometry_mismatches = []
        centres, radii = [], []
        physical_centres, physical_radii = [], []
        for o in self.obstacles:
            h = o.height if o.shape == "cylinder" else o.size_z
            rad = (math.hypot(o.radius, h/2) if o.shape == "cylinder" else
                   math.sqrt(o.size_x**2 + o.size_y**2 + h**2)/2)
            if o.shape == "sphere":
                rad = o.radius
            physical_centres.append([o.x, o.y, o.z])
            physical_radii.append(rad)
            if o.shape not in ("cylinder", "box"):
                self.geometry_mismatches.append(dict(name=o.name, issue="renderer_unsupported_shape"))
                continue
            # This deliberately reproduces the archived renderer's bounds;
            # it does NOT silently correct its grounded geometry assumption.
            centres.append([o.x, o.y, h/2])
            radii.append(rad)
            if abs(o.z-h/2) > .001:
                self.geometry_mismatches.append(dict(name=o.name, issue="renderer_grounds_shape",
                                                     physical_z=o.z, renderer_z=h/2))
            if o.shape == "box" and abs(math.sin(2*o.yaw)) > .001 and abs(o.size_x-o.size_y) > .001:
                self.geometry_mismatches.append(dict(name=o.name, issue="renderer_box_yaw_sign"))
        self.centres = np.asarray(centres).reshape(-1, 3)
        self.radii = np.asarray(radii)
        self.physical_centres = np.asarray(physical_centres).reshape(-1, 3)
        self.physical_radii = np.asarray(physical_radii)

    def clear(self, pos, goal):
        for o in self.obstacles:
            if o.shape == "sphere":
                a = np.asarray(pos)-[o.x, o.y, o.z]
                d = np.asarray(goal)-pos
                t = np.clip(-np.dot(a, d)/max(np.dot(d, d), 1e-20), 0, 1)
                if np.linalg.norm(a+t*d) <= o.radius+.1:
                    return False
        return segment_obstacle_collision(pos, goal, self.obstacles, .1) is None

    def range_empty(self, pos):
        # Includes camera's 0.121 m mount offset. Sufficient, not necessary.
        return bool(np.all(np.linalg.norm(self.physical_centres-pos, axis=1)
                           - self.physical_radii > 10.122))

    def initial_blocker_plane(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        axis = (goal-start)/np.linalg.norm(goal-start)
        far = None
        for o in self.obstacles:
            one = Scene([vars(o)])
            if one.clear(start, goal):continue
            if o.shape == 'cylinder':
                extent = o.radius*np.linalg.norm(axis[:2])+o.height/2*abs(axis[2])
            elif o.shape == 'box':
                c, s = math.cos(o.yaw), math.sin(o.yaw)
                extent = (o.size_x/2*abs(axis[0]*c+axis[1]*s)
                          +o.size_y/2*abs(-axis[0]*s+axis[1]*c)+o.size_z/2*abs(axis[2]))
            else:extent = o.radius
            edge = float(np.dot(np.array([o.x,o.y,o.z])-start, axis)+extent+.1)
            far = edge if far is None else max(far, edge)
        return axis, far

    def certified_empty_view(self, pos, yaw):
        if not len(self.centres):
            return True
        c, s = math.cos(yaw), math.sin(yaw)
        origin = np.asarray(pos) + [.12*c, .12*s, .015]
        v = self.centres-origin
        xb, yb, zb = v[:, 0]*c+v[:, 1]*s, -v[:, 0]*s+v[:, 1]*c, v[:, 2]
        cp, sp = math.cos(math.radians(8)), math.sin(math.radians(8))
        x, y, z = -yb, -sp*xb-cp*zb, cp*xb-sp*zb
        kx, ky = math.tan(1.047198/2), .75*math.tan(1.047198/2)
        r = self.radii
        possible = (np.linalg.norm(v, axis=1)-r <= 10.) & (z+r >= 0)
        for q, k in [(x, kx), (y, ky)]:
            possible &= (k*z+q >= -r*math.sqrt(1+k*k)) & (k*z-q >= -r*math.sqrt(1+k*k))
        return not bool(np.any(possible))


def angle_error(pos, goal, yaw):
    v = np.asarray(goal)-pos
    return abs(math.atan2(math.sin(math.atan2(v[1], v[0])-yaw),
                          math.cos(math.atan2(v[1], v[0])-yaw))) * 180/math.pi


def episode(header, rows, end, scene, task_number):
    goal = np.asarray(header['task']['goal_pos'])
    start = np.asarray(header['task']['start_pos'])
    axis, far = scene.initial_blocker_plane(start, goal)
    last_three_empty = None
    cycles = []
    for row in rows:
        pos = np.asarray(row['pre_state']['position'])
        after = np.asarray(row['post_state']['position'])
        yaw = row['pre_state']['yaw']
        empty = scene.certified_empty_view(pos, yaw)
        if last_three_empty is None:
            last_three_empty = [empty]*3
        meta = row['pointcloud_metadata_before'][-3:]
        ground = empty and meta[-1]['valid_hit_rays'] > 0
        history_ground = all(last_three_empty) and all(m['valid_hit_rays'] > 0 for m in meta)
        d, dn = float(np.linalg.norm(goal-pos)), float(np.linalg.norm(goal-after))
        path = np.asarray(row['executed_path'])
        cycles.append(dict(decision=row['decision'], clear=scene.clear(pos, goal),
                           beyond_initial_blockers=far is not None and float(np.dot(pos-start, axis))>far,
                           distance=d, progress=d-dn, yaw_error_deg=angle_error(pos, goal, yaw),
                           empty_view_certificate=empty, ground_view_certificate=ground,
                           ground_history_certificate=history_ground,
                           latest_fallback=meta[-1]['used_far_plane_fallback'],
                           any_history_fallback=any(m['used_far_plane_fallback'] for m in meta),
                           all_history_fallback=all(m['used_far_plane_fallback'] for m in meta),
                           hit_fraction=1-meta[-1]['no_ray_hit_fraction'],
                           finite=all(m['finite'] for m in meta),
                           range_empty=scene.range_empty(pos),
                           path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())))
        # Nonterminal rows have all four frames. The last frame uses post_state.
        yaws = yaw+np.cumsum(np.asarray(row['executed_action_4x11'])[:, 9])
        poses = path[1:]
        flags = [scene.certified_empty_view(p, float(yaws[i])) for i, p in enumerate(poses)]
        if flags:
            flags[-1] = scene.certified_empty_view(after, row['post_state']['yaw'])
            last_three_empty = (last_three_empty+flags)[-3:]
    clear = [c['clear'] for c in cycles]
    rs = runs(clear)
    sustained = [(a, b) for a, b in rs if b-a >= 3]
    first = sustained[0][0] if sustained else None
    final_d = float(np.linalg.norm(goal-np.asarray(rows[-1]['post_state']['position'])))
    free_segments = []
    for a, b in sustained:
        selected = cycles[a:b]
        free_segments.append(dict(start=a, end_exclusive=b, decisions=b-a,
            start_distance=cycles[a]['distance'], progress=sum(c['progress'] for c in selected),
            ground_history_cycles=sum(c['ground_history_certificate'] for c in selected),
            latest_fallback_cycles=sum(c['latest_fallback'] for c in selected),
            path_length=sum(c['path_length'] for c in selected)))
    stalled = []
    for a, b in sustained:
        for i in range(a, b-4):
            window = cycles[i:i+5]
            if sum(c['progress'] for c in window) <= .1:
                stalled.append(dict(start=i, ground_history_cycles=sum(c['ground_history_certificate'] for c in window),
                                    path_length=sum(c['path_length'] for c in window),
                                    progress=sum(c['progress'] for c in window)))
    blocked_stalls = [i for i in range(len(cycles)-4)
                      if all(not c['clear'] for c in cycles[i:i+5])
                      and sum(c['progress'] for c in cycles[i:i+5])<=.1]
    positions = np.asarray([row['pre_state']['position'] for row in rows])
    lengths = np.r_[0.,np.cumsum([c['path_length'] for c in cycles])]
    revisits = []
    for j in range(5,len(positions)):
        for i in range(j-4):
            if lengths[j]-lengths[i]>2 and np.linalg.norm(positions[j]-positions[i])<=.5:
                revisits.append(dict(start=i,end=j,all_blocked=not any(clear[i:j+1])))
                break
    passed = [i for i,c in enumerate(cycles) if c['beyond_initial_blockers'] and c['clear']]
    return dict(policy=header['policy'], task_id=header['task_id'], task=task_number,
                world=header['world_id'], repeat=header['repeat'], success=end['safe_arrival'],
                outcome=end['done_reason'], phase=phase_label(clear), initial_clear=clear[0],
                decisions=len(cycles), first_sustained_clear=first,
                first_sustained_clear_distance=None if first is None else cycles[first]['distance'],
                final_distance=final_d, final_clear_run=rs[-1] if rs and rs[-1][1]==len(clear) else None,
                first_passed_initial_blockers=passed[0] if passed else None,
                distance_after_passing=None if not passed else cycles[passed[0]]['distance'],
                goal_distance_beyond_blocker_plane=None if far is None else float(np.linalg.norm(goal-start)-far),
                blocked_stall_windows=blocked_stalls, revisit_candidates=revisits,
                clear_segments=free_segments, clear_stalled_windows=stalled,
                cycles=cycles)


def summarize(records):
    result = {}
    for policy in sorted(set(r['policy'] for r in records)):
        rr = [r for r in records if r['policy']==policy]
        summary = dict(n=len(rr), outcomes=dict(Counter(r['outcome'] for r in rr)),
                       phases={}, cycle_groups={}, success_clear_start_distance={}, timeout_flags={})
        for phase in sorted(set(r['phase'] for r in rr)):
            chosen = [r for r in rr if r['phase']==phase]
            summary['phases'][phase] = dict(n=len(chosen), outcomes=dict(Counter(r['outcome'] for r in chosen)))
        groups = defaultdict(list)
        for r in rr:
            for c in r['cycles']:
                groups['all'].append(c)
                if c['clear']:
                    groups['clear'].append(c)
                    groups['clear_ground_history' if c['ground_history_certificate'] else 'clear_other_view'].append(c)
                    if c['distance']>2: groups['clear_distance_gt2'].append(c)
                if c['ground_history_certificate']:groups['ground_history'].append(c)
                if c['any_history_fallback']:groups['any_history_fallback'].append(c)
        for group, cc in groups.items():
            summary['cycle_groups'][group] = dict(n=len(cc),
                mean_progress=float(np.mean([c['progress'] for c in cc])),
                approach_gt05=sum(c['progress']>.05 for c in cc),
                away_lt_minus05=sum(c['progress']<-.05 for c in cc),
                mean_yaw_error=float(np.mean([c['yaw_error_deg'] for c in cc])),
                yaw_error_gt90=sum(c['yaw_error_deg']>90 for c in cc),
                latest_fallback=sum(c['latest_fallback'] for c in cc),
                all_history_fallback=sum(c['all_history_fallback'] for c in cc),
                nonfinite=sum(not c['finite'] for c in cc),
                ground_history=sum(c['ground_history_certificate'] for c in cc),
                range_empty=sum(c['range_empty'] for c in cc))
        for initial in [False, True]:
            chosen = [r for r in rr if r['success'] and r['initial_clear']==initial]
            bins = Counter()
            for r in chosen:
                d = r['first_sustained_clear_distance']
                bins['no_3decision_clear_run' if d is None else 'le2m' if d<=2 else '2to5m' if d<=5 else 'gt5m'] += 1
            summary['success_clear_start_distance']['initially_clear' if initial else 'initially_blocked'] = dict(bins)
        timed = [r for r in rr if r['outcome']=='time_limit']
        summary['timeout_flags'] = dict(n=len(timed),
            any_ground_history=sum(any(c['ground_history_certificate'] for c in r['cycles']) for r in timed),
            any_fallback=sum(any(c['any_history_fallback'] for c in r['cycles']) for r in timed),
            clear_stall_5cycles=sum(bool(r['clear_stalled_windows']) for r in timed),
            clear_ground_stall_5cycles=sum(any(w['ground_history_cycles']==5 for w in r['clear_stalled_windows']) for r in timed),
            clear_final_10cycles=sum(r['final_clear_run'] is not None and r['final_clear_run'][1]-r['final_clear_run'][0]>=10 for r in timed))
        result[policy] = summary
    return result


def training_analysis(root, scenes):
    counts = defaultdict(Counter)
    by_world = defaultdict(Counter)
    n = 0
    missing = Counter()
    for path in sorted(root.glob('metrics_update_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        u = int(path.stem.split('_')[-1])
        window = '1-32' if u<=32 else '33-128' if u<=128 else '129-256' if u<=256 else '257-384'
        for t in data['rollout_audit']['transition_records']:
            scene = scenes[t['world_id']]
            clear = scene.clear(t['position_before_xyz'], t['goal_xyz'])
            empty = scene.range_empty(t['position_before_xyz'])
            keys = ['all', 'clear' if clear else 'blocked']
            if empty:keys.append('range_empty_all_directions')
            if clear and empty:keys.append('clear_and_range_empty')
            for key in keys:
                c = counts[(window, key)]
                c['n'] += 1
                c['approach_gt05'] += t['net_goal_approach_3d']>.05
                c['away_lt_minus05'] += t['net_goal_approach_3d']<-.05
                c['progress_sum'] += t['net_goal_approach_3d']
                c['raw_adv_positive'] += t['advantage_raw']>0
                c['actor_adv_positive'] += t['dppo_credit']['advantage_normalized_for_update']>0
                c['newbest_positive'] += t['reward_terms']['newbest']>0
                c['reached'] += t['terminal_facts']['reached']
                c['distance_sum'] += t['goal_distance_before_3d']
            c = by_world[(window, t['world_id'])]
            c['n'] += 1
            c['clear'] += clear
            c['range_empty_all_directions'] += empty
            n += 1
            missing['yaw'] += 'yaw' not in t and 'yaw_before' not in t
            missing['pointcloud_metadata'] += 'pointcloud_metadata_before' not in t
        if u%32==0:print('training update',u,flush=True)
    def convert(c):
        c = dict(c)
        for key in ['progress', 'distance']:
            if key+'_sum' in c:c['mean_'+key] = c.pop(key+'_sum')/c['n']
        return c
    result = dict(n=n, missing_fields=dict(missing), windows={}, by_world={})
    for (w, k), c in counts.items():result['windows'].setdefault(w, {})[k] = convert(c)
    for (w, k), c in by_world.items():result['by_world'].setdefault(w, {})[k] = dict(c)
    # Warnings are an occurrence lower bound, not a frame or transition count.
    warnings = Counter()
    pending = 0
    for line in (root/'train.log').open(encoding='utf-8'):
        if 'RuntimeWarning: synth: no ray hit' in line:pending += 1
        if line.startswith('{'):
            try:obj=json.loads(line)
            except json.JSONDecodeError:continue
            if obj.get('event')=='dppo_outer_update_complete':
                u=obj['update_index']
                w='1-32' if u<=32 else '33-128' if u<=128 else '129-256' if u<=256 else '257-384'
                warnings[w]+=pending
                pending=0
    result['logged_nohit_warnings_lower_bound'] = dict(warnings)
    result['unassigned_log_warnings'] = pending
    return result


def replanning_episode(previous):
    from diffusion_policy_3d.env.traj_utils import action_to_world
    from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
    trace = Path(previous['source'])
    data = trace.read_bytes()
    objects = [json.loads(line) for line in data.splitlines()]
    header, rows, end = objects[0], objects[1:-1], objects[-1]
    assert len(rows)==previous['decisions'] and end['done_reason']==previous['outcome']
    goal = np.asarray(header['task']['goal_pos'])
    start = np.asarray(header['task']['start_pos'])
    axis = (goal-start)/np.linalg.norm(goal-start)
    scene = Scene(header['geometry'])
    cycles, paths = [], []
    closest = np.linalg.norm(goal-start)
    for index, row in enumerate(rows):
        assert row['type']=='transition' and row['decision']==index
        pre = row['pre_state']
        pos = np.asarray(pre['position'], np.float32)
        full = np.asarray(row['full_action_16x11'], np.float32)
        assert np.array_equal(full[2:10], np.asarray(row['policy_action_8x11'], np.float32))
        world = action_to_world(full[2:], pos, np.asarray(pre['velocity'], np.float32), pre['yaw'])
        path = np.vstack((pos, world[:, :3]))
        assert np.allclose(path[:9], row['planned_policy_path'], atol=1e-6, rtol=0)
        f = plan_features(path, goal, world[:, 9], pre['velocity'], pre['yaw'])
        base = previous['cycles'][index]
        f.update({k:base[k] for k in ('decision','clear','ground_history_certificate',
            'beyond_initial_blockers','distance','progress','yaw_error_deg')})
        f['past_goal_plane'] = bool(np.dot(pos-goal, axis)>0)
        f['previously_within_2m'] = bool(closest<=2)
        f['nonterminal'] = row['done_reason']=='none'
        f['h8_path_safe'] = None
        f['safe_h8_tail_goal'] = False
        # Geometry certifies this entire candidate; it is not a flown outcome.
        needs_safety = ((f['tail_progress']>.1 and f['h4_progress']<=.05)
                        or (f['approach_onset'] is not None and 5<=f['approach_onset']<=8))
        if needs_safety:
            f['h8_path_safe'] = bool(np.min(path[:9, 2])>=.2 and
                                     all(scene.clear(a,b) for a,b in zip(path[:8],path[1:9])))
        a, delta = path[:8]-goal, np.diff(path[:9], axis=0)
        den = np.sum(delta*delta, axis=1)
        t = np.clip(np.divide(-np.sum(a*delta,axis=1), den,
                             out=np.zeros(8), where=den>1e-20),0,1)
        if np.min(np.linalg.norm(a+t[:,None]*delta,axis=1))<=.500001 and f['nonterminal']:
            event = locate_first_event(path[:9], goal, scene.obstacles, .1,
                header['task']['z_range'][1], terminate_at_space_limits=False)
            f['safe_h8_tail_goal'] = bool(event is not None and event.kind=='reached'
                                         and event.segment_index>=4)
        f['delayed_approach'] = bool(f['nonterminal'] and f['clear'] and
            f['approach_onset'] is not None and 5<=f['approach_onset']<=8 and
            f['h4_progress']<=.05 and f['tail_progress']>.1 and f['h8_path_safe'])
        executed = np.asarray(row['executed_path'])
        aa, dd = executed[:-1]-goal, np.diff(executed,axis=0)
        den = np.sum(dd*dd,axis=1)
        t = np.clip(np.divide(-np.sum(aa*dd,axis=1),den,
                             out=np.zeros(len(den)),where=den>1e-20),0,1)
        closest = min(closest, float(np.min(np.linalg.norm(aa+t[:,None]*dd,axis=1))))
        cycles.append(f)
        paths.append(path)
    pairs = []
    for i in range(len(rows)-1):
        f, following = cycles[i], cycles[i+1]
        q = compare_replans(paths[i], paths[i+1], rows[i]['post_state']['position'], goal)
        q['decision'] = i
        q['clear_both'] = f['clear'] and following['clear']
        q['ground_both'] = f['ground_history_certificate'] and following['ground_history_certificate']
        q['old_tail_safe'] = f['h8_path_safe']
        q['tail_promise'] = bool(q['clear_both'] and f['h8_path_safe'] and
                                f['h4_progress']<=.05 and q['old_tail_progress']>.1)
        q['promise_cancelled'] = q['tail_promise'] and q['new_prefix_progress']<=.05
        q['repeated_delay_pair'] = f['delayed_approach'] and following['delayed_approach']
        q['actual_approach_then_reversal'] = bool(q['clear_both'] and f['progress']>.1 and
                                                 following['progress']<-.1)
        q['new_prefix_still_approaches'] = q['tail_promise'] and q['new_prefix_progress']>.1
        pairs.append(q)
    flags = dict(repeated_delay_3=any(b-a>=3 for a,b in runs([c['delayed_approach'] for c in cycles])),
        any_delayed_approach=any(c['delayed_approach'] for c in cycles),
        any_cancelled_promise=any(p['promise_cancelled'] for p in pairs),
        any_actual_reversal=any(p['actual_approach_then_reversal'] for p in pairs),
        safe_h8_tail_goal=any(c['safe_h8_tail_goal'] for c in cycles))
    result = {k:previous[k] for k in ('policy','task','task_id','repeat','outcome','success','source')}
    result.update(trace_sha256=hashlib.sha256(data).hexdigest(), cycles=cycles, pairs=pairs,
        flags=flags, repeated_delay_runs=runs([c['delayed_approach'] for c in cycles]))
    return result


def replanning_main(cache_path, output_path):
    started = time.monotonic()
    cache = json.loads(cache_path.read_text(encoding='utf-8'))
    records, totals = [], defaultdict(Counter)
    last_policy = None
    for previous in cache['records']:
        policy = previous['policy']
        if last_policy is not None and policy!=last_policy:
            print('replanning',last_policy,'episodes',len(records),'seconds',round(time.monotonic()-started,1),flush=True)
        last_policy = policy
        r = replanning_episode(previous)
        records.append(r)
        for group in ('all', r['outcome']):
            c = totals[(policy,group)]
            c['episodes'] += 1
            c.update({k:int(v) for k,v in r['flags'].items()})
            c['cycles'] += len(r['cycles'])
            c['pairs'] += len(r['pairs'])
            for q in r['pairs']:
                for k in ('tail_promise','promise_cancelled','repeated_delay_pair',
                          'actual_approach_then_reversal','new_prefix_still_approaches'):
                    c[k+'_pairs'] += int(q[k])
            for f in r['cycles']:
                c['clear_cycles'] += int(f['clear'])
                c['delayed_cycles'] += int(f['delayed_approach'])
    summary = {}
    for (policy,group),c in totals.items():summary.setdefault(policy,{})[group]=dict(c)
    output = dict(date='2026-10-17', scope='offline_replanning_of_existing_v3_traces',
        behavior_cache=str(cache_path), n=len(records),
        definitions=dict(indices='one-based future segments: H4=1..4 H8=1..8 full[2:]=1..14',
            approach='two consecutive segments each reduce 3D goal distance >0.01m; sensitivity 0.005 and 0.02m',
            delayed='nonterminal, direct corridor clear, first approach in 5..8, H4 net <=0.05m, tail net >0.1m, whole H8 physically clear and z>=0.2m',
            repeated='at least three consecutive delayed decisions; observational pattern, not causal attribution',
            overlap='old planned waypoints 5..8 versus next planned waypoints 1..4 in common world coordinates',
            tail_promise='both corridors clear, whole old H8 safe, old H4 net<=0.05m and old tail net>0.1m',
            cancelled='tail promise followed by new H4 net<=0.05m',
            actual_reversal='two clear decisions, actual first net>0.1m then next net<-0.1m',
            beyond8='generated suffix only, outside current H8 risk/credit window; not executed evidence'),
        summary=summary, records=records, elapsed_seconds=time.monotonic()-started,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(output,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('saved',output_path,'episodes',len(records),'seconds',round(output['elapsed_seconds'],1),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-dir', type=Path)
    p.add_argument('--existing-analysis', type=Path)
    p.add_argument('--replanning-cache', type=Path,
                   help='Analyze raw H16/H8/H4 replanning, reusing existing geometry/visibility labels')
    p.add_argument('--training-dir', type=Path)
    p.add_argument('--training-cache', type=Path,
                   help='Reuse the unchanged training section from an earlier diagnostic output')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.replanning_cache:
        replanning_main(args.replanning_cache, args.output)
        return
    if args.full_dir is None or args.existing_analysis is None:
        p.error('--full-dir and --existing-analysis are required without --replanning-cache')
    cached_training = (json.loads(args.training_cache.read_text(encoding='utf-8'))['training']
                       if args.training_cache else None)
    started = time.monotonic()
    previous = json.loads(args.existing_analysis.read_text(encoding='utf-8'))
    task_map = {r['task_id']:r['task'] for r in previous['records']}
    previous_blocked = {(r['policy'], r['task_id'], r['repeat']):r['blocked'] for r in previous['records']}
    scenes, records = {}, []
    for job in sorted(args.full_dir.iterdir()):
        if not (job/'commits').is_dir():continue
        for commit in sorted((job/'commits').glob('*.json')):
            ref = json.loads(commit.read_text(encoding='utf-8'))
            trace = job/ref['trace']
            with trace.open(encoding='utf-8') as stream:
                h = json.loads(next(stream))
                raw = [json.loads(line) for line in stream]
            rows = [r for r in raw if r['type']=='transition']
            end = raw[-1]
            assert len(rows)==ref['summary']['decisions']
            scene = scenes.setdefault(h['world_id'], Scene(h['geometry']))
            record = episode(h, rows, end, scene, task_map[h['task_id']])
            assert (not record['initial_clear'])==previous_blocked[(h['policy'], h['task_id'], h['repeat'])]
            record['source'] = str(trace)
            records.append(record)
        print('evaluation',job.name, len(records), 'elapsed',round(time.monotonic()-started,1), flush=True)
    output = dict(date='2026-10-17', scope='existing_lightweight_kinematic_traces',
                  definitions=dict(clear='continuous current-position to goal segment using physical collision proxy',
                    sustained_clear='at least 3 consecutive pre-action decision states',
                    empty_view='sufficient bounding-sphere exclusion from archived renderer camera frustum and 10m range',
                    ground_history='all latest 3 frame views certified obstacle-free, each with >0 recorded valid rays; pre-noise ground source',
                    range_empty='all physical obstacle bounding spheres >10.122m from drone; sufficient for any camera yaw',
                    stall='5 clear pre-action decisions with summed net goal approach <=0.1m; descriptive, not intent',
                    warnings='lower bound of emitted no-hit warnings; no unique frame denominator'),
                  n=len(records), cycles=sum(r['decisions'] for r in records),
                  summary=summarize(records),
                  geometry_mismatches={w:s.geometry_mismatches for w,s in scenes.items() if s.geometry_mismatches},
                  records=records)
    if args.training_dir:
        output['training'] = training_analysis(args.training_dir, scenes)
    elif cached_training is not None:
        output['training'] = cached_training
        output['training_reused_from'] = str(args.training_cache)
    output['elapsed_seconds'] = time.monotonic()-started
    output['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('saved',args.output,'episodes',output['n'],'cycles',output['cycles'],flush=True)


if __name__ == '__main__':
    main()
