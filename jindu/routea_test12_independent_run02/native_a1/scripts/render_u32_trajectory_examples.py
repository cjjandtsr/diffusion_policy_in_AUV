#!/usr/bin/env python3
"""Render three representative BC/RL XY trajectory comparisons as SVG."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dppo_u32_trajectory_repeats import parse_trace_csv, resolve_cli_path


CASES = (
    ("seed3", "env_016_obs18__cfg03", "Seed3: BC success -> RL timeout"),
    ("seed1", "env_012_obs22__cfg01", "Seed1: collision -> rescue"),
    ("seed3", "routeA_rl_v1__dense_recovery__rl_test__s_turn_multi_obstacle__000047__cfg03",
     "Seed3: dense S-turn rescue"),
)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#8A8A8A"


def _geometry_path(world: str, run_dir: Path, layera: Path, rl_test: Path) -> Path:
    request = json.loads((run_dir / "replay_request.json").read_text(encoding="utf-8"))
    source = request["resolved_inputs"]["obstacle_sources"][world]
    return (layera if source["split"] == "layerA" else rl_test) / Path(source["path"]).name


def _obstacle_extent(obstacle: dict) -> tuple[float, float, float, float]:
    if obstacle.get("shape", "cylinder") == "cylinder":
        radius = float(obstacle["radius"])
        return (float(obstacle["x"]) - radius, float(obstacle["x"]) + radius,
                float(obstacle["y"]) - radius, float(obstacle["y"]) + radius)
    half_x, half_y = float(obstacle["size_x"]) / 2, float(obstacle["size_y"]) / 2
    radius = math.hypot(half_x, half_y)
    return (float(obstacle["x"]) - radius, float(obstacle["x"]) + radius,
            float(obstacle["y"]) - radius, float(obstacle["y"]) + radius)


def _box_points(obstacle: dict, sx, sy) -> str:
    hx, hy = float(obstacle["size_x"]) / 2, float(obstacle["size_y"]) / 2
    yaw, cx, cy = float(obstacle.get("yaw", 0)), float(obstacle["x"]), float(obstacle["y"])
    c, s = math.cos(yaw), math.sin(yaw)
    points = []
    for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        wx, wy = cx + c * x - s * y, cy + s * x + c * y
        points.append(f"{sx(wx):.1f},{sy(wy):.1f}")
    return " ".join(points)


def render(traces_root: Path, episode_csv: Path, layera: Path, rl_test: Path, output: Path) -> None:
    with episode_csv.open("r", encoding="utf-8", newline="") as handle:
        episodes = list(csv.DictReader(handle))
    run_dirs = {seed: next(path for path in traces_root.iterdir() if path.is_dir() and f"_{seed}_v2" in path.name)
                for seed, _, _ in CASES}
    width, height = 1500, 540
    panel_w, plot_h = 480, 400
    left, top = 55, 72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#222} .tick{font-size:11px;fill:#555} '
        '.title{font-size:16px;font-weight:600}.small{font-size:12px;fill:#444}</style>',
        '<text x="24" y="28" class="title">Route A u32: paired executed trajectories (five repeats)</text>',
        f'<line x1="950" y1="24" x2="985" y2="24" stroke="{BLUE}" stroke-width="3"/><text x="991" y="28" class="small">BC</text>',
        f'<line x1="1045" y1="24" x2="1080" y2="24" stroke="{ORANGE}" stroke-width="3"/><text x="1086" y="28" class="small">RL</text>',
        f'<rect x="1140" y="16" width="14" height="14" fill="{GREY}" opacity="0.35"/><text x="1160" y="28" class="small">obstacle footprint</text>',
    ]
    for panel, (seed, task_id, title) in enumerate(CASES):
        rows = [row for row in episodes if row["seed"] == seed and row["task_id"] == task_id]
        traces = [parse_trace_csv(run_dirs[seed] / row["trajectory_file"]) for row in rows]
        world = str(traces[0].header["world"])
        obstacles = json.loads(_geometry_path(world, run_dirs[seed], layera, rl_test).read_text(encoding="utf-8"))["obstacles"]
        points = np.vstack([trace.executed[:, 3:6] for trace in traces])
        start = np.asarray(traces[0].header["start"], dtype=float)
        goal = np.asarray(traces[0].header["goal"], dtype=float)
        x_values = [*points[:, 0], start[0], goal[0]]
        y_values = [*points[:, 1], start[1], goal[1]]
        for obstacle in obstacles:
            xmin, xmax, ymin, ymax = _obstacle_extent(obstacle)
            x_values.extend((xmin, xmax)); y_values.extend((ymin, ymax))
        xmin, xmax = min(x_values), max(x_values)
        ymin, ymax = min(y_values), max(y_values)
        span = max(xmax - xmin, ymax - ymin) * 1.08
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        xmin, xmax, ymin, ymax = cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2
        x0 = panel * panel_w + left
        y0 = top
        plot_w = 390
        sx = lambda value, x0=x0: x0 + (value - xmin) / (xmax - xmin) * plot_w
        sy = lambda value, y0=y0: y0 + plot_h - (value - ymin) / (ymax - ymin) * plot_h
        parts.append(f'<text x="{x0}" y="52" class="title">{html.escape(title)}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#555"/>')
        for obstacle in obstacles:
            if obstacle.get("shape", "cylinder") == "cylinder":
                radius_px = float(obstacle["radius"]) / (xmax - xmin) * plot_w
                parts.append(f'<circle cx="{sx(float(obstacle["x"])):.1f}" cy="{sy(float(obstacle["y"])):.1f}" r="{radius_px:.1f}" fill="{GREY}" opacity="0.35" stroke="#666"/>')
            else:
                parts.append(f'<polygon points="{_box_points(obstacle, sx, sy)}" fill="{GREY}" opacity="0.35" stroke="#666"/>')
        outcomes: dict[str, Counter] = {"bc": Counter(), "rl": Counter()}
        for trace, row in zip(traces, rows):
            policy = row["policy"]
            outcomes[policy][row["done"]] += 1
            xy = trace.executed[:, 3:5]
            polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in xy)
            colour = BLUE if policy == "bc" else ORANGE
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{colour}" stroke-width="2" opacity="0.55"/>')
        parts.append(f'<circle cx="{sx(start[0]):.1f}" cy="{sy(start[1]):.1f}" r="5" fill="#009E73" stroke="#222"/>')
        parts.append(f'<circle cx="{sx(goal[0]):.1f}" cy="{sy(goal[1]):.1f}" r="7" fill="none" stroke="#222" stroke-width="2"/>')
        parts.append(f'<text x="{x0 + plot_w / 2}" y="{y0 + plot_h + 34}" text-anchor="middle" class="small">x world (m)</text>')
        parts.append(f'<text x="{x0 - 38}" y="{y0 + plot_h / 2}" text-anchor="middle" transform="rotate(-90 {x0 - 38} {y0 + plot_h / 2})" class="small">y world (m)</text>')
        label = lambda counter: ", ".join(f"{key} {value}/5" for key, value in sorted(counter.items()))
        parts.append(f'<text x="{x0}" y="{y0 + plot_h + 56}" class="small" fill="{BLUE}">BC: {html.escape(label(outcomes["bc"]))}</text>')
        parts.append(f'<text x="{x0 + 190}" y="{y0 + plot_h + 56}" class="small" fill="{ORANGE}">RL: {html.escape(label(outcomes["rl"]))}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", required=True, type=Path)
    parser.add_argument("--episode-csv", required=True, type=Path)
    parser.add_argument("--layera-geometry", required=True, type=Path)
    parser.add_argument("--rl-test-geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(*(resolve_cli_path(value) for value in (
        args.traces_root, args.episode_csv, args.layera_geometry, args.rl_test_geometry, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
