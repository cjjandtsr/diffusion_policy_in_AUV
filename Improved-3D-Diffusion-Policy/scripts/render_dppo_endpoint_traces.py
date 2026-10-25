#!/usr/bin/env python3
"""Render verified paired traces; no policy/model access or new sampling."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnose_dppo_uav_endpoint import read_json, sha256, write_json


def read_verified_traces(source):
    source = Path(source).resolve()
    verification = read_json(source / "replay_verification.json")
    checks = verification["checks"]
    if (verification["status"] != "matched" or len(checks) != verification["completed_episodes"]
            or len(checks) != 2 * verification["selected_pairs"]):
        raise ValueError("only complete matched replay packs can be rendered")
    pairs = {}
    for check in checks:
        path = (source / check["trace_file"]).resolve()
        if path.parent != source or not check["matched"] or sha256(path) != check["trace_sha256"]:
            raise ValueError("trace hash/status/path mismatch")
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not events or events[-1]["event"] != "episode_end" or events[-1]["reproduction_differences"]:
            raise ValueError("trace has no matching terminal record")
        if any(e["task_id"] != check["task_id"] or e["policy"] != check["policy"] for e in events):
            raise ValueError("trace pair identity mismatch")
        pair = pairs.setdefault(check["task_id"], {})
        if check["policy"] in pair:
            raise ValueError("duplicate trace policy")
        task_path = source / (path.name.split(".")[0] + ".task.json")
        if sha256(task_path) != check["task_sha256"]:
            raise ValueError("task geometry hash mismatch")
        task = read_json(task_path)
        if task["task_id"] != check["task_id"]:
            raise ValueError("geometry task identity mismatch")
        pair[check["policy"]] = (events, task)
    if any(set(pair) != {"bc", "rl"} for pair in pairs.values()):
        raise ValueError("unpaired traces")
    return pairs


def obstacle_wireframes(geometry, *, sensor=False):
    """Draw collision solids by default; optionally show actual sensor solids."""
    from dataclasses import asdict
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec
    for obstacle in geometry.get("obstacles", []):
        if not sensor:
            obstacle = asdict(ObstacleSpec.from_dict(obstacle))
        kind = obstacle.get("shape", "cylinder")
        if kind == "cylinder":
            theta = np.linspace(0., 2 * np.pi, 33)
            xy = np.column_stack((np.cos(theta), np.sin(theta))) * obstacle["radius"]
            height = obstacle["height"]
        elif kind == "box":
            xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
            xy = xy * [obstacle.get("size_x", 2.) / 2, obstacle.get("size_y", 2.) / 2]
            yaw = obstacle.get("yaw", 0.)
            xy = xy @ np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            height = obstacle.get("size_z", 4.)
        else:
            raise ValueError("unsupported obstacle shape: " + kind)
        xy += [obstacle["x"], obstacle["y"]]
        low = 0. if sensor else obstacle["z"] - height / 2
        bottom = np.column_stack((xy, np.full(len(xy), low)))
        top = np.column_stack((xy, np.full(len(xy), low + height)))
        yield bottom
        yield top
        for index in (range(0, 32, 8) if kind == "cylinder" else range(4)):
            yield np.vstack((bottom[index], top[index]))


def evaluated_trap_state(step):
    # Episode-end history cleanup is not evidence of recovery on this step.
    return step["reward_metrics"]["trap_state"]


def render_pair(task_id, pair, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(16, 12), layout="constrained")
    grid = fig.add_gridspec(3, 4)
    spatial = [fig.add_subplot(grid[0, 0], projection="3d"),
               fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]
    info_ax = fig.add_subplot(grid[0, 3])
    info_ax.axis("off")
    spatial[0].set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Executed path / world frame")
    spatial[1].set(xlabel="x (m)", ylabel="y (m)", title="XY")
    spatial[2].set(xlabel="x (m)", ylabel="z (m)", title="XZ (obstacle projection)")
    axes = [fig.add_subplot(grid[r, c]) for r in (1, 2) for c in range(4)]
    labels = ["Goal distance (m)", "Altitude (m)", "Executed length / nominal time (m/s)",
              "Minimum geometric clearance (m)", "Reward", "True hit fraction per PC frame",
              "Trap state", "High-altitude stall seconds per step"]
    for ax, label in zip(axes, labels):
        ax.set(xlabel="Nominal execution time (s)", title=label)
        ax.grid(alpha=.2)
    geometry = pair["bc"][1]["geometry"]
    collision_lines = list(obstacle_wireframes(geometry))
    sensor_lines = list(obstacle_wireframes(geometry, sensor=True))
    geometry_differs = any(not np.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(collision_lines, sensor_lines))
    for line in collision_lines:
        spatial[0].plot(*line.T, color="gray", alpha=.2, linewidth=.6)
        spatial[1].plot(line[:, 0], line[:, 1], color="gray", alpha=.2, linewidth=.6)
        spatial[2].plot(line[:, 0], line[:, 2], color="gray", alpha=.2, linewidth=.6)
    if geometry_differs:
        for line in sensor_lines:
            spatial[0].plot(*line.T, color="red", alpha=.25, linewidth=.6, linestyle="--")
            spatial[1].plot(line[:, 0], line[:, 1], color="red", alpha=.25, linewidth=.6, linestyle="--")
            spatial[2].plot(line[:, 0], line[:, 2], color="red", alpha=.25, linewidth=.6, linestyle="--")
    descriptions, reward_data = [], []
    for policy, color in (("bc", "tab:blue"), ("rl", "tab:orange")):
        events, task = pair[policy]
        reset = next(e for e in events if e["event"] == "reset")
        steps = [e for e in events if e["event"] == "transition"]
        if not steps or any(e["executed_path"] is None for e in steps):
            raise ValueError("missing executed path; cannot fabricate a trajectory")
        start, goal = np.asarray(reset["state"]["position"]), np.asarray(reset["state"]["goal"])
        path = np.vstack([start] + [np.asarray(s["executed_path"])[1:] for s in steps])
        spatial[0].plot(*path.T, color=color, label=policy.upper())
        spatial[0].scatter(*start, color="black", marker="o", s=20)
        spatial[0].scatter(*goal, color="green", marker="*", s=80)
        for ax, coordinates in zip(spatial[1:], ((0, 1), (0, 2))):
            ax.plot(path[:, coordinates[0]], path[:, coordinates[1]], color=color, label=policy.upper())
            ax.scatter(goal[coordinates[0]], goal[coordinates[1]], color="green", marker="*")
            ax.set_aspect("equal", adjustable="datalim")
        times = [s["time_end_seconds"] for s in steps]
        series = [[s["after"]["distance_to_goal"] for s in steps],
                  [s["after"]["position"][2] for s in steps],
                  [np.linalg.norm(np.diff(s["executed_path"], axis=0), axis=1).sum() / 1.2 for s in steps],
                  [s["reward_metrics"].get("d_min") for s in steps], [s["reward"] for s in steps]]
        for ax, values in zip(axes[:5], series):
            ax.plot(times, values, color=color, label=policy.upper())
        frames = [e for e in events if e["event"] == "pointcloud"]
        frame_times = [0. if f["phase"] == "reset" else (f["decision_step"]-1)*1.2+(f["frame_in_step"]+1)*.3 for f in frames]
        axes[5].plot(frame_times, [f["hit_count"] / f["ray_count"] for f in frames], color=color)
        trap_states = {"FREE": 0, "CANDIDATE": 1, "TRAPPED": 2, "RECOVERING": 3}
        axes[6].step(times, [trap_states[evaluated_trap_state(s)] for s in steps], color=color)
        axes[6].set(yticks=list(trap_states.values()), yticklabels=list(trap_states))
        axes[7].plot(times, [s["reward_metrics"].get("above_demo_stall_seconds") for s in steps], color=color)
        summary = events[-1]["summary"]
        descriptions.append("{}: {}\nsteps={} return={:.3f}\ngoal tolerance: distance<{}, z<{}".format(
            policy.upper(), summary["done_reason"], summary["steps"], summary["episode_return"],
            reset["goal_tol_distance"], reset["goal_tol_z"]))
        reward_data.append((policy, steps, times))
    descriptions.append("Gray: obstacle collision solids (unexpanded).\n" +
                        ("RED DASHES: sensor geometry differs!" if geometry_differs else "Sensor/collision solid coordinates agree."))
    info_ax.text(0, 1, "\n\n".join(descriptions), va="top", fontsize=9)
    spatial[0].legend()
    axes[0].legend()
    fig.suptitle(task_id + "\nDiagnostic subset; reproduced summaries; lightweight H4, not Gazebo/PX4", fontsize=11)
    stem = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:20]
    fig.savefig(output / (stem + ".png"), dpi=130)
    plt.close(fig)
    names = sorted(reward_data[0][1][0]["reward_terms"])
    matrices = [np.array([[s["reward_terms"][name] for s in steps] for name in names]) for _, steps, _ in reward_data]
    limit = max(max(abs(m).max() for m in matrices), 1e-6)
    fig, reward_axes = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")
    for ax, (policy, steps, times), matrix in zip(reward_axes, reward_data, matrices):
        im = ax.imshow(matrix, aspect="auto", interpolation="none", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), extent=(0, times[-1], len(names)-.5, -.5))
        ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Nominal execution time (s)", title=policy.upper())
        fig.colorbar(im, ax=ax, label="Per-step reward contribution; shared scale")
    fig.suptitle(task_id + " / all reward terms")
    fig.savefig(output / (stem + ".rewards.png"), dpi=130)
    plt.close(fig)
    return dict(task_id=task_id, trajectory_plot=stem + ".png", reward_plot=stem + ".rewards.png",
                sensor_collision_geometry_differs=geometry_differs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs = read_verified_traces(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index = [render_pair(task, pair, args.output_dir) for task, pair in sorted(pairs.items())]
    write_json(args.output_dir / "figure_index.json", dict(pair_count=len(index), figures=index,
               verification_sha256=sha256(args.source_dir / "replay_verification.json")))


if __name__ == "__main__":
    main()
