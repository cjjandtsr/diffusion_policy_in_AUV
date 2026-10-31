"""Preregistered geometric counterexamples for Reward v1.7 calibration."""

import math

import numpy as np


CONTEXT = dict(ground_z=0., xy_min=[-5., -5.], xy_max=[5., 5.])


def line(start, end):
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return [(a + (b - a) * q).tolist() for q in (0., .25, .5, .75, 1.)]


def episode(eid, paths, *, start, goal, obstacles=(), outcome="reached", role="fixture"):
    return dict(episode_id=eid, world_id="fixture_v17", role=role, split="fixture",
                start=list(start), goal=list(goal), initial_yaw=0.,
                obstacles=list(obstacles), geometry_context=dict(CONTEXT),
                steps=[dict(executed_path=p, end_yaw=0.) for p in paths],
                expected_done_reason=outcome)


def paths_from_endpoints(start, endpoints):
    paths, current = [], list(start)
    for end in endpoints:
        paths.append(line(current, end))
        current = list(end)
    return paths


def make_fixture_manifest(source_path, source_sha256):
    start, goal = [-4., 0., 1.], [4., 0., 1.]
    wall = dict(name="full_width_wall", shape="box", x=0., y=0., z=1., yaw=0.,
                size_x=.4, size_y=10., size_z=2.)
    over = [[start, [-1., 0., 2.2], [0., 0., 2.2], [1., 0., 2.2], goal]]
    straight = [line(start, goal)]
    ground = [[start, [-2., 0., .5], [-1., 0., -.1], [-.5, 0., .5], [-.25, 0., 1.]]]
    timeout = [line(start, start) for _ in range(60)]
    episodes = [
        episode("wall_overflight", over, start=start, goal=goal, obstacles=[wall]),
        episode("wall_collision", straight, start=start, goal=goal, obstacles=[wall], outcome="collision"),
        episode("wall_ground", ground, start=start, goal=goal, obstacles=[wall], outcome="collision"),
        episode("wall_timeout", timeout, start=start, goal=goal, obstacles=[wall], outcome="time_limit"),
    ]

    roof_obstacle = dict(name="roof_box", shape="box", x=0., y=0., z=1., yaw=0.,
                         size_x=.4, size_y=1., size_z=2.)
    roof_start, roof_goal = [-2., 0., 2.15], [2., 0., 2.15]
    roof_side = [[roof_start, [-1., 1.2, 2.15], [0., 1.2, 2.15], [1., 1.2, 2.15], roof_goal]]
    roof_low = [[roof_start, [-1., 0., 2.15], [0., 0., 2.15], [1., 0., 2.15], roof_goal]]
    episodes += [
        episode("roof_safe_side", roof_side, start=roof_start, goal=roof_goal, obstacles=[roof_obstacle]),
        episode("roof_clearance_5cm", roof_low, start=roof_start, goal=roof_goal, obstacles=[roof_obstacle]),
    ]

    cylinder = dict(name="central_cylinder", shape="cylinder", x=0., y=0., z=1., yaw=0.,
                    radius=.3, height=2.)
    side_safe = [[start, [-2., 1.2, 1.], [0., 1.2, 1.], [2., 1.2, 1.], goal]]
    episodes += [
        episode("side_route_safe", side_safe, start=start, goal=goal, obstacles=[cylinder]),
        episode("side_route_collision", straight, start=start, goal=goal, obstacles=[cylinder], outcome="collision"),
    ]

    direct_start, direct_goal = [0., 0., 1.], [4., 0., 1.]
    direct = [line(direct_start, direct_goal)]
    efficiency_start, efficiency_goal = [-2., -1.5, 1.], [2., -1.5, 1.]
    efficiency_obstacle = dict(name="efficiency_box", shape="box", x=0., y=0., z=1., yaw=0.,
                               size_x=.4, size_y=.4, size_z=2.)
    efficient = [line(efficiency_start, efficiency_goal)]
    # The direct route safely skirts the box.  The redundant route enters the
    # same horizontal footprint well above its roof, so both climb and the
    # obstacle-relative high-overflight feature are strictly positive.
    redundant = [[efficiency_start, [-1., 0., 3.], [0., 0., 4.],
                  [1., 0., 3.], efficiency_goal]]
    episodes += [
        episode("efficient_direct", efficient, start=efficiency_start, goal=efficiency_goal,
                obstacles=[efficiency_obstacle]),
        episode("redundant_high", redundant, start=efficiency_start, goal=efficiency_goal,
                obstacles=[efficiency_obstacle]),
    ]

    high_start, high_goal = [0., 0., 1.], [0., 0., 5.5]
    high_direct_ends = [[0., 0., z] for z in (1.7, 2.4, 3.1, 3.8, 4.6, 5.5)]
    # The first full six-step window and the three following windows end in
    # distinct above-demo cells.  This makes the support-gaming witness test
    # the episode escrow itself, rather than merely comparing two max heights.
    high_detour_ends = [[x, 0., z] for x, z in ((1., 1.8), (2., 2.8), (3., 4.),
                                                 (3., 5.2), (3., 6.), (2.5, 6.),
                                                 (2., 6.), (1., 6.), (0., 5.5))]
    episodes += [
        episode("high_goal_direct", paths_from_endpoints(high_start, high_direct_ends),
                start=high_start, goal=high_goal, obstacles=[]),
        episode("high_goal_support_detour", paths_from_endpoints(high_start, high_detour_ends),
                start=high_start, goal=high_goal, obstacles=[]),
    ]

    loop_endpoints = [[.3, 0., 1.], [0., 0., 1.]] * 4 + [[4., 0., 1.]]
    episodes += [
        episode("clear_loop_reference", direct, start=direct_start, goal=direct_goal, obstacles=[]),
        episode("clear_loop", paths_from_endpoints(direct_start, loop_endpoints),
                start=direct_start, goal=direct_goal, obstacles=[]),
    ]

    trap_obstacles = [
        dict(name="trap_near_1", shape="cylinder", x=-.7, y=0., z=1., yaw=0., radius=.1, height=2.),
        dict(name="trap_near_2", shape="cylinder", x=2.4, y=-1., z=1., yaw=0., radius=.1, height=2.),
    ]
    trap_start, trap_goal = [0., 0., 1.], [4., 1., 1.]
    # Seven stationary transitions enter segment 1.  The next three moves
    # create two consecutive full-window recoveries with clearance above the
    # entry clearance plus epsilon_d, returning the state machine to FREE.
    # Moving near the second cylinder and waiting seven transitions then
    # enters a distinct segment 2 before the final safe reach.
    second_trap = [2.4, -.55, 1.]
    trap_ends = [trap_start] * 7
    trap_ends += [[.8, -.7, 1.], [1.4, -.4, 1.], [1.8, 0., 1.], second_trap]
    trap_ends += [second_trap] * 6 + [trap_goal]
    episodes += [
        episode("trap_reference", [line(trap_start, trap_goal)], start=trap_start, goal=trap_goal,
                obstacles=trap_obstacles),
        episode("repeated_trap", paths_from_endpoints(trap_start, trap_ends), start=trap_start,
                goal=trap_goal, obstacles=trap_obstacles),
    ]

    # Additional successful obstacle-relative routes make the fixture anchor
    # clearance distribution non-degenerate without duplicating observations.
    anchor_start, anchor_goal = [-2., 0., 1.], [2., 0., 1.]
    anchor_obstacle = dict(name="anchor_cylinder", shape="cylinder", x=0., y=0., z=1., yaw=0.,
                           radius=.2, height=2.)
    for index, lateral in enumerate((.45, .55, .7, .85, 1., 1.2, 1.45, 1.7)):
        path = [[anchor_start, [-1., lateral, 1.], [0., lateral, 1.],
                 [1., lateral, 1.], anchor_goal]]
        episodes.append(episode("anchor_{:02d}".format(index), path, start=anchor_start,
                                goal=anchor_goal, obstacles=[anchor_obstacle], role="anchor"))

    categories = {
        "necessary_overflight_vs_collision": [["wall_overflight", "wall_collision"]],
        "necessary_overflight_vs_ground": [["wall_overflight", "wall_ground"]],
        "necessary_overflight_vs_timeout": [["wall_overflight", "wall_timeout"]],
        "safe_vs_roof_5cm": [["roof_safe_side", "roof_clearance_5cm"]],
        "side_route_vs_failure": [["side_route_safe", "side_route_collision"]],
        "overflight_vs_failure": [["wall_overflight", "wall_collision"]],
        "safe_vs_redundant_high": [["efficient_direct", "redundant_high"]],
        "high_goal_direct_vs_support_detour": [["high_goal_direct", "high_goal_support_detour"]],
        "safe_vs_clear_loop": [["clear_loop_reference", "clear_loop"]],
        "safe_vs_repeated_trap": [["trap_reference", "repeated_trap"]],
    }
    comparisons = []
    for pairs in categories.values():
        for pair in pairs:
            if pair not in comparisons:
                comparisons.append(pair)
    return dict(
        schema_version="reward_v17_fixture_manifest_v1", calibration_source="fixture",
        source_evidence=dict(
            files=[dict(path=source_path, sha256=source_sha256)],
            coordinate_reference=dict(frame="fixture_world_z_up", ground_z=0.,
                                      generation="deterministic literal geometry from Reward v1.7 v1.4")),
        ordering_categories=categories, comparisons=comparisons, episodes=episodes,
    )
