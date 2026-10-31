"""run_adapter_regression_gate.py — replay the G4 fixture and emit the
structured gate report (design §5.6 / Guide R4.1 Step 4).

Replays every fixture case through the canonical UAVVectorEnvAdapter
boundary and compares against the oracle outputs frozen in the fixture:

- admission determinism: the adapter reset with the fixture's (spec, seed)
  must resolve the EXACT recorded episode config (start/goal/yaw/id);
- per step: position / yaw / vel (allclose rtol=1e-5 atol=1e-6 — the
  design §5.6 recorded threshold), agent_pos (8,9), reward, termination
  flags, action command (exact), point-cloud frames (np.allclose at the
  design threshold for hit frames — the frozen CUDA synth pipeline is
  not bitwise-stable run-to-run, so the design's own allclose criterion
  is enforced with per-frame max_abs_diff recorded; no-ray frames skip
  the pixel comparison by design — R2.5 replaced the frozen all-far
  fallback with a physical far-endpoint encoding, so those frames
  legitimately differ and are compared only on position/state);
- the candidate must not terminate earlier or later than the fixture.

The report binds the fixture file by sha256 and records provenance
(command, exit code, tree hash, source hashes). An existing report is
verified against the fixture BEFORE the replay — a tampered or
regenerated fixture hard-fails the gate.

Run:
  python scripts/run_adapter_regression_gate.py \
      --fixture /home/cjj/AUV/program/reports/routeA_rl_env_v1/\
g4_adapter_regression/ADAPTER_REGRESSION_FIXTURE.json \
      --output /home/cjj/AUV/program/reports/routeA_rl_env_v1/\
g4_adapter_regression/GAZEBO_KINEMATIC_REGRESSION_REPORT.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSpec,
    UAVVectorEnvAdapter,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.integration import build_adapter

DEFAULT_FIXTURE = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/g4_adapter_regression"
    "/ADAPTER_REGRESSION_FIXTURE.json"
)
DEFAULT_OUTPUT = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/g4_adapter_regression"
    "/GAZEBO_KINEMATIC_REGRESSION_REPORT.json"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"

RTOL = 1e-5
ATOL = 1e-6


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(frame, dtype=np.float32).tobytes()
    ).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _git_tree() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _allclose_msg(name: str, got: np.ndarray, want: np.ndarray) -> str:
    return (
        f"{name} deviates: max_abs_diff="
        f"{float(np.abs(np.asarray(got) - np.asarray(want)).max()):.3e}"
        f" (rtol={RTOL}, atol={ATOL})"
    )


def _verify_reset_cfg(rec: Dict[str, Any], case: Dict[str, Any]) -> List[str]:
    """The adapter's resolved episode config must equal the fixture's
    recorded initial state exactly (admission determinism)."""
    want = case["initial_state"]
    problems: List[str] = []
    if rec.get("world") != case["world_uid"]:
        problems.append(
            f"world resolved {rec.get('world')!r} != fixture "
            f"{case['world_uid']!r}"
        )
    if rec.get("seed") != case["seed"]:
        problems.append(
            f"seed resolved {rec.get('seed')!r} != fixture {case['seed']!r}"
        )
    if rec.get("start_yaw") != want["start_yaw"]:
        problems.append(
            f"start_yaw resolved {rec.get('start_yaw')!r} != fixture "
            f"{want['start_yaw']!r}"
        )
    if rec.get("episode_id") != want["episode_config"]["episode_id"]:
        problems.append(
            f"episode_id resolved {rec.get('episode_id')!r} != fixture "
            f"{want['episode_config']['episode_id']!r}"
        )
    for key in ("start", "goal"):
        if not np.array_equal(
            np.asarray(rec[key], dtype=np.float32),
            np.asarray(want[key], dtype=np.float32),
        ):
            problems.append(
                f"{key} resolved {rec[key]} != fixture {want[key]}"
            )
    return problems


def _fixture_frame(want: Dict[str, Any], i: int) -> np.ndarray:
    """Load one expected frame from the fixture's .npy sidecar and verify
    its recorded content hash (tamper/staleness gate)."""
    path = Path(want["frame_files"][i])
    frame = np.ascontiguousarray(np.load(path), dtype=np.float32)
    recorded = want["pc_frame_sha256"][i]
    if _frame_sha256(frame) != recorded:
        raise SystemExit(
            f"fixture frame {path} content hash mismatch — the fixture "
            "was tampered with or its sidecars were overwritten"
        )
    return frame


def _compare_obs(
    obs: Dict[str, np.ndarray], want: Dict[str, Any], adapter, label: str,
) -> Tuple[List[str], List[str], List[float]]:
    """agent_pos + point-cloud comparison against the fixture record.

    Returns (problems, notes, pc_max_diffs). Point clouds are compared at
    the design §5.6 recorded threshold (np.allclose rtol=1e-5 atol=1e-6),
    NOT bitwise: the frozen CUDA synth pipeline (depth noise + dropout) is
    not bitwise-stable run-to-run (measured 60/60 repeat-run ULP
    deviations, max 1.6e-5 abs on ~10-magnitude points; the CPU synth is
    0/60 — the design's own acceptance criterion is the allclose
    threshold, so the gate enforces exactly that). No-ray frames
    legitimately differ (R2.5 validated-synth far-endpoint encoding) and
    are recorded as a note.
    """
    problems: List[str] = []
    notes: List[str] = []
    pc_max_diffs: List[float] = []
    poses = obs["agent_pos"][0]
    want_poses = np.asarray(want["agent_pos"], dtype=np.float32)
    if poses.shape != want_poses.shape:
        problems.append(
            f"{label} agent_pos shape {poses.shape} != {want_poses.shape}"
        )
    elif not np.allclose(poses, want_poses, rtol=RTOL, atol=ATOL):
        problems.append(_allclose_msg(f"{label} agent_pos", poses, want_poses))
    pcs = obs["point_cloud"][0]
    for i in range(pcs.shape[0]):
        live_hits = int(adapter._hit_count(pcs[i]))
        want_hits = int(want["hit_count"][i])
        if live_hits != want_hits:
            problems.append(
                f"{label} frame {i} hit_count {live_hits} != "
                f"fixture {want_hits}"
            )
            pc_max_diffs.append(float("nan"))
            continue
        if live_hits == 0:
            notes.append(
                f"{label} frame {i}: no-ray — pc skipped by design (R2.5 "
                "far-endpoint encoding), position/state compared"
            )
            pc_max_diffs.append(float("nan"))
            continue
        want_pc = _fixture_frame(want, i)
        diff = float(
            np.abs(pcs[i].astype(np.float64) - want_pc.astype(np.float64)).max()
        )
        pc_max_diffs.append(diff)
        if not np.allclose(pcs[i], want_pc, rtol=RTOL, atol=ATOL):
            problems.append(
                f"{label} frame {i} pc deviates: max_abs_diff={diff:.3e}"
                f" (rtol={RTOL}, atol={ATOL})"
            )
    return problems, notes, pc_max_diffs


def _replay_case(
    fixture: Dict[str, Any], case: Dict[str, Any],
    worlds_dir: str, manifest: str,
) -> Dict[str, Any]:
    """Replay one fixture case through the adapter boundary."""
    problems: List[str] = []
    notes: List[str] = []

    contract = RLSystemContract()
    spec = EpisodeTaskSpec(**case["spec"])
    records = {
        manifest_record_key(spec.world_uid, spec.role, spec.admission_id):
            spec.manifest_entry_hash
    }
    adapter = build_adapter(
        contract,
        worlds_dir=worlds_dir,
        manifest_path=manifest,
        manifest_records=records,
        device=fixture["candidate_device"],
        max_episode_steps=60,
        n_envs=1,
    )

    seed = case["seed"]
    obs, _ = adapter.reset_at([0], [spec], [seed])
    rec = adapter.current_episode_config(0)
    problems += _verify_reset_cfg(rec, case)
    init = case["initial_state"]["reset_obs"]
    p, n, reset_diffs = _compare_obs(obs, init, adapter, "reset")
    problems += p
    notes += n
    pc_max_diffs: Dict[str, List[float]] = {"reset": reset_diffs}

    # ── action-command spy: the adapter must feed env.step exactly the
    #    fixture action (float32 cast is the boundary's documented
    #    conversion, so the comparison is against the cast array) ──
    env = adapter._envs[0]
    orig_step = env.step
    commands: List[np.ndarray] = []

    def spy_step(action, *args, **kwargs):
        commands.append(np.asarray(action).copy())
        return orig_step(action, *args, **kwargs)

    env.step = spy_step

    actions = np.asarray(case["actions"], dtype=np.float32)
    expected = case["expected"]
    for k, want in enumerate(expected):
        batch = adapter.step(actions[k][None])
        if len(commands) != k + 1:
            problems.append(f"step {k}: env.step not reached via adapter")
            break
        cmd = commands[-1]
        if not np.array_equal(cmd, np.asarray(want["action_command"])):
            problems.append(
                f"step {k} action command != fixture actions "
                f"(max_abs_diff={float(np.abs(cmd - actions).max()):.3e})"
            )
        obs_k = {
            "agent_pos": batch.obs["agent_pos"],
            "point_cloud": batch.obs["point_cloud"],
        }
        p, n, step_diffs = _compare_obs(obs_k, want, adapter, f"step {k}")
        problems += p
        notes += n
        pc_max_diffs[f"step_{k}"] = step_diffs
        state = env._drone_pos
        if not np.allclose(state, np.asarray(want["position"]),
                           rtol=RTOL, atol=ATOL):
            problems.append(
                _allclose_msg(f"step {k} position", state, want["position"])
            )
        if not np.isclose(float(env._drone_yaw), float(want["yaw"]),
                          rtol=RTOL, atol=ATOL):
            problems.append(
                f"step {k} yaw {env._drone_yaw} != fixture {want['yaw']}"
            )
        if not np.allclose(env._drone_vel, np.asarray(want["vel"]),
                           rtol=RTOL, atol=ATOL):
            problems.append(
                _allclose_msg(f"step {k} vel", env._drone_vel, want["vel"])
            )
        if not np.isclose(float(batch.reward[0]), float(want["reward"]),
                          rtol=RTOL, atol=ATOL):
            problems.append(
                f"step {k} reward {batch.reward[0]} != fixture "
                f"{want['reward']}"
            )
        if bool(batch.terminated[0]) != bool(want["terminated"]):
            problems.append(
                f"step {k} terminated {bool(batch.terminated[0])} != "
                f"fixture {want['terminated']}"
            )
        if bool(batch.truncated[0]) != bool(want["truncated"]):
            problems.append(
                f"step {k} truncated {bool(batch.truncated[0])} != "
                f"fixture {want['truncated']}"
            )
        if (bool(batch.terminated[0]) or bool(batch.truncated[0])) \
                and k != len(expected) - 1:
            problems.append(
                f"step {k} terminated but fixture expects "
                f"{len(expected)} steps (early stop mismatch)"
            )
            break

    # candidate must not outlive the fixture's executed steps
    if len(expected) < int(case["steps_planned"]) and not (
        expected[-1]["terminated"] or expected[-1]["truncated"]
    ):
        problems.append(
            "fixture stopped early without a termination flag — fixture "
            "inconsistent"
        )
    return {
        "case_id": case["case_id"],
        "world_uid": case["world_uid"],
        "seed": seed,
        "pc_max_abs_diff": pc_max_diffs,
        "problems": problems,
        "notes": notes,
        "status": "PASS" if not problems else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--worlds-dir", default=DEFAULT_WORLDS)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required: the frozen E040 path synthesizes on cuda:0"
        )

    fixture_path = Path(args.fixture)
    if not fixture_path.is_file():
        raise SystemExit(f"fixture not found: {fixture_path}")
    fixture = json.loads(fixture_path.read_text())
    if fixture["fixture_type"] != "replacement_adapter_non_regression_fixture":
        raise SystemExit(f"unexpected fixture_type: {fixture['fixture_type']}")
    assert fixture["tolerance"] == {"rtol": RTOL, "atol": ATOL}, (
        "fixture tolerance differs from the design §5.6 recorded threshold"
    )

    fixture_sha = _sha256(fixture_path)
    report_path = Path(args.output)
    if report_path.is_file():
        old = json.loads(report_path.read_text())
        if old["fixture_sha256"] != fixture_sha:
            raise SystemExit(
                "fixture changed since the last gate report — rebuild the "
                "fixture and re-run the gate (no silent overwrite)"
            )

    results = [
        _replay_case(fixture, case, args.worlds_dir, args.manifest)
        for case in fixture["cases"]
    ]
    all_pass = all(r["status"] == "PASS" for r in results)

    report = {
        "gate": "G4",
        "gate_source": (
            "design §5.6 + Guide R4.1; replacement fixture replay via "
            "scripts/run_adapter_regression_gate.py"
        ),
        "replacement_label": "replacement_adapter_non_regression_fixture",
        "equivalence_fact": fixture["equivalence_fact"],
        "tolerance": fixture["tolerance"],
        "fixture": fixture_path.name,
        "fixture_sha256": fixture_sha,
        "oracle_source_hashes": fixture["oracle"]["source_files"],
        "candidate_source_hashes": fixture["candidate"]["source_files"],
        "oracle_device": fixture["oracle_device"],
        "candidate_device": fixture["candidate_device"],
        "created_commit": fixture["created_commit"],
        "project_git_commit": _git_head(),
        "project_tree_hash": _git_tree(),
        "command": " ".join(sys.argv),
        "results": results,
        "status": "PASS" if all_pass else "FAIL",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
