"""Descriptive paired tables and raw executed-trajectory plots, no verdict."""
import collections
from itertools import combinations
from pathlib import Path

import numpy as np

from .core import ROLES, paired_summary, write_json
from .episodes import load_commit


def grouped(records, fields):
    groups = collections.defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        entry = dict(zip(fields, key))
        counts = collections.Counter(r["done_reason"] for r in rows)
        entry.update({"episodes": len(rows), "tasks": len({r["task_id"] for r in rows}),
                      "worlds": len({r["world_id"] for r in rows}),
                      "safe_arrival": sum(bool(r["safe_arrival"]) for r in rows),
                      "safe_arrival_rate": sum(bool(r["safe_arrival"]) for r in rows) / len(rows)})
        entry["ambiguous_first_event_episodes"] = sum(bool(r.get("ambiguous_first_event")) for r in rows)
        for kind in ("collision", "ground_crash", "out_of_bounds", "altitude_exceed", "time_limit"):
            entry[kind] = counts[kind]
        for metric in rows[0]["trajectory_metrics"]:
            entry["mean_" + metric] = float(np.mean([r["trajectory_metrics"][metric] for r in rows]))
        entry["mean_physical_time_s"] = float(np.mean([r["physical_time_s"] for r in rows]))
        result.append(entry)
    return result


def tables(records):
    policies = sorted({r["policy"] for r in records})
    comparisons = []
    for reference, treatment in combinations(policies, 2):
        comparisons.append({"overall_descriptive": paired_summary(records, treatment, reference),
                            "by_role": {role: paired_summary([r for r in records if r["role"] == role],
                                                             treatment, reference) for role in ROLES
                                        if any(r['role'] == role for r in records)}})
    episode_rows = []
    for row in records:
        episode_rows.append({k: row[k] for k in ("policy", "task_id", "world_id", "role", "repeat",
                                                  "status", "safe_arrival", "done_reason", "decisions",
                                                  "physical_time_s", "initial_hash")})
        episode_rows[-1].update(row["trajectory_metrics"])
        episode_rows[-1]["ambiguous_first_event"] = bool(row.get("ambiguous_first_event"))
    return {"episode_metrics": episode_rows,
            "task_policy": grouped(records, ("task_id", "world_id", "role", "policy")),
            "world_policy": grouped(records, ("world_id", "role", "policy")),
            "role_policy": grouped(records, ("role", "policy")),
            "comparisons": comparisons, "policies": policies,
            "scope": "training-domain development; roles describe source tasks, not proven capabilities",
            "paper_effect_threshold_frozen": False, "paper_conclusion": "not determined",
            "BC_is_one_checkpoint_with_shared_evaluation_repeats": True,
            "pooled_success_rate_is_not_the_sole_decision": True}


def plot_obstacles(ax, obstacles):
    for o in obstacles:
        if o["shape"] in ("cylinder", "sphere"):
            angles = np.linspace(0, 2*np.pi, 33)
            if o["shape"] == "cylinder":
                for z in (o["z"]-o["height"]/2, o["z"]+o["height"]/2):
                    ax.plot(o["x"]+o["radius"]*np.cos(angles), o["y"]+o["radius"]*np.sin(angles),
                            np.full_like(angles, z), color="gray", alpha=.45, linewidth=.7)
                for a in angles[::8]:
                    ax.plot([o["x"]+o["radius"]*np.cos(a)]*2, [o["y"]+o["radius"]*np.sin(a)]*2,
                            [o["z"]-o["height"]/2,o["z"]+o["height"]/2], color="gray", alpha=.45, linewidth=.7)
            else:
                ax.plot(o["x"]+o["radius"]*np.cos(angles), o["y"]+o["radius"]*np.sin(angles),
                        np.full_like(angles, o["z"]), color="gray", alpha=.5)
        elif o["shape"] == "box":
            xy = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]], float)
            xy *= [o["size_x"]/2,o["size_y"]/2]
            c,s = np.cos(o["yaw"]),np.sin(o["yaw"])
            xy = xy @ np.array([[c,s],[-s,c]]) + [o["x"],o["y"]]
            for z in (o["z"]-o["size_z"]/2,o["z"]+o["size_z"]/2):
                ax.plot(xy[:,0],xy[:,1],np.full(5,z),color="gray",alpha=.5,linewidth=.7)
            for x,y in xy[:4]:
                ax.plot([x,x],[y,y],[o["z"]-o["size_z"]/2,o["z"]+o["size_z"]/2],
                        color="gray",alpha=.5,linewidth=.7)


def plot_tasks(bank, run_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .bank import episode_key
    out = Path(out)
    out.mkdir()
    jobs = (run_dir if isinstance(run_dir, dict) else
            {p.name: p for p in Path(run_dir).iterdir() if (p / 'commits').is_dir()})
    policies = sorted(jobs, key=lambda name: (name != 'bc', name))
    for index, task in enumerate(bank["tasks"]):
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        world = bank["worlds"][task["world_id"]]
        plot_obstacles(ax, world["geometry"])
        all_points = []
        for n, policy in enumerate(policies):
            job = Path(jobs[policy])
            key = episode_key(task["task_id"],1)
            header, rows, summary = load_commit(job, job/"commits"/(key+".json"))
            points = [header["task"]["start_pos"]]
            for row in rows:
                points.extend(row["executed_path"][1:])
            p = np.asarray(points)
            all_points.extend(p)
            colour = 'black' if policy == 'bc' else plt.cm.tab10(n % 10)
            style = "-" if n==0 else ("-","--",":")[(n-1)%3]
            ax.plot(p[:,0],p[:,1],p[:,2],color=colour,linestyle=style,
                    linewidth=1.8 if n==0 else 1.2,
                    label=policy+" "+summary["done_reason"])
            ax.scatter(*p[-1],color=colour,marker="o" if summary["safe_arrival"] else "x",s=18)
        goal = np.asarray(task["payload"]["goal_pos"])
        ax.scatter(*task["payload"]["start_pos"],color="green",s=35,label="start")
        u,v = np.meshgrid(np.linspace(0,2*np.pi,24),np.linspace(0,np.pi,12))
        ax.plot_wireframe(goal[0]+.5*np.cos(u)*np.sin(v),goal[1]+.5*np.sin(u)*np.sin(v),
                          goal[2]+.5*np.cos(v),color="purple",alpha=.4,linewidth=.5)
        ax.scatter(*goal,color="purple",s=30,label="goal sphere r=0.5 m")
        p = np.vstack((np.asarray(all_points),goal))
        ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),1.))
        ax.set_xlabel("world x / m"); ax.set_ylabel("world y / m"); ax.set_zlabel("world z / m")
        ax.set_title(task["role"]+" | task %d | repeat 1 (fixed, not selected for outcome)"%index)
        ax.legend(loc="upper left",fontsize=7)
        fig.tight_layout()
        fig.savefig(out/("task_%03d.png"%index),dpi=150)
        plt.close(fig)
    write_json(out/"figure_index.json", {"fixed_repeat":1,"tasks":[t["task_id"] for t in bank["tasks"]],
                                       "selection":"all frozen tasks; no outcome-based selection"})
