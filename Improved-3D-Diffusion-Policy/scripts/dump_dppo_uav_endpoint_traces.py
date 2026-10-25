#!/usr/bin/env python3
"""Save the full per-decision-step trajectory of one frozen endpoint cell.

Minimal variant of scripts/replay_dppo_uav_endpoint_traces.py. Only two
differences: every one of the 128 pairs is replayed (no diagnostic subset),
and the checkpoint update is taken from the frozen provenance instead of a
hard-coded 256. All provenance pinning and the reproduction check are
unchanged, so a trace file is only written when the replayed episode
reproduces the original summary to rtol/atol 1e-6.

Each transition additionally carries the identity of the input the policy
consumed for that decision (see endpoint_trajectory_trace_v2.PolicyRecorder),
and each trace header carries its position in the evaluation order. That is
what localises a BC/RL divergence to the pipeline or to the policy: under a
shared initial condition the two policies must agree on observation and
policy-input hashes until their physics diverges, so a hash difference before
divergence is a pipeline fault, while identical inputs with a differing
`full_action` is the policy.

Coordinate frames. Each task file and every trace line carries a `frames` block
declaring which frame each position-bearing field is in, plus that world's
`spawn_offset`. `agent_pos` is odom while `position` / `executed_path` / `goal`
are world, so they differ by exactly that offset -- a difference that otherwise
reads as a pipeline fault in the comparison above. The offset is read through
`KinematicUAVEnv._load_spawn_offset`, the same reader the env uses, so the block
cannot drift from the env's actual convention.

Trajectory tables. Each episode writes one `.csv` holding two tables: the
policy's planned trajectory (8 world-frame waypoints per decision step) and the
executed trajectory (positions only, one row per time step). The name is
`<world>__x<..>_y<..>_z<..>__<policy>__r<repeat>`, and a `#` header repeats the
world, split, policy, start, goal, repeat and `spawn_offset`, so a trajectory
identifies itself without any other file. Layout in
`endpoint_trajectory_trace_v2.write_trajectory_csv`.

Repeats. `--repeats N` replays every (world, start, goal) cell N times per
selected policy, so one cell contributes N samples and a single draw's noise
does not decide a comparison. Repeat 1 replays the seeds recorded in the bank
and is checked against the original episode exactly as before. Both recorded
seeds are pure functions of the config hash (`repeat_seeds`), so replaying them
again would reproduce run 1 bit for bit and add no sample; repeats 2..N
therefore take a fresh env reset seed and a shifted action seed derived from the
same hash. A later repeat is a different draw, so its differences from the
recorded episode are recorded in `replay_verification.json`, not treated as a
replay fault.

`--artifacts` selects how much beside the trajectory is kept. `trajectory` (the
default) writes only the CSV plus `replay_request.json` and
`replay_verification.json`. `full` additionally writes one `.task.json` (world
geometry, fixed config) and one `.trace.jsonl` per policy, the per-decision event
stream. Both modes run the same input resolution, the same reproduction check
and the same plan-versus-executed assertion, so the trajectory is equally
trustworthy either way; `full` only adds the per-decision event record.

Two ways in, and the difference is whether there is a recorded episode to check
against. `--source-dir` replays a recorded endpoint pack: its provenance fixes
the config, the checkpoint, the episode banks and each world's obstacle, and the
first repeat of every cell is checked against the episode recorded there. That
check is the point of a replay. Without `--source-dir` the same inputs are
resolved from `--config`, `--rl-checkpoint` and `--eval-config-dir` exactly as
`evaluate_dppo_uav_formal_endpoint.run` resolves them, and each selected world's
obstacle from `--worlds-test` / `--worlds-layera`. No record then exists, so
nothing is compared: that mode collects new trajectory samples rather than
reproducing old ones, and `replay_verification.json` reports `not_compared`
instead of a comparison it did not make. `--limit N` keeps the first N cells in
bank order, so a smoke run covers the cells a full run starts with.

Where the output goes. With no `--output-dir` the run lands at
`<shared root>/traces_<benchmark>/<YYYYMMDD_HHMMSS>_<label>/`, where the shared
root is `program/shiyan/rl_bc_endpoint_comparison` -- the one home for every
RL-versus-BC endpoint artifact, see `scripts/endpoint_output_root.py`. The label
is the source pack's directory name, or the config's file name in pack-free
mode. An explicit `--output-dir` still wins, and a path outside that root prints
a warning so a stray collection cannot pass unnoticed.

This is policy inference, not training. No production file is modified.
"""

import argparse
import contextlib
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_dppo_uav_endpoint import load_evidence, read_json, sha256, write_json
from scripts.endpoint_output_root import (
    DEFAULT_ROOT, outside_default_root, resolve_output_dir,
)
from scripts.endpoint_trajectory_trace_v2 import (
    PolicyRecorder, TraceAdapter, compare_record, frame_contract,
    write_trajectory_csv,
)
from scripts.evaluate_dppo_uav_formal_endpoint import (
    BENCHMARK_SPLITS, _build_adapter, _E5Policy, _evaluate_one, _load_fixed_bank,
    _resolve_obstacle, _rl_policy_identity, _validate_bank_sources,
)


def bound_file(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError("source asset SHA256 mismatch: " + str(path))
    return path


def trajectory_stem(world_id, start_pos, policy_name, repeat):
    """`<world>__x<..>_y<..>_z<..>__<policy>__r<repeat>`.

    The world name already carries split and role, so this names one trajectory
    completely. Two decimals is enough: no world in the four episode banks has
    two configs whose starts agree at two decimals. The caller checks for
    collisions anyway, so a future bank cannot silently overwrite a file.
    """
    def fixed(value):
        text = "{:.2f}".format(float(value))
        return "0.00" if text == "-0.00" else text
    x, y, z = (fixed(value) for value in start_pos)
    return "{}__x{}_y{}_z{}__{}__r{}".format(world_id, x, y, z, policy_name, repeat + 1)


def repeat_seeds(config_hash, repeat):
    """The env reset seed and action-seed offset of one repeat.

    `canonical_episode_config` derives both seeds from the config hash
    (`evaluate_dppo_uav_formal_endpoint.py:92-95`), so replaying a repeat with
    the recorded values would reproduce the first run bit for bit and add no
    sample. Repeat 0 keeps them untouched -- it is the reproduction run --
    and every later repeat gets a seed derived the same way, so a repeat is a
    fresh draw of both the env's reset noise and the policy's sampling.
    """
    digest = hashlib.sha256(
        "{}:repeat:{}".format(config_hash, repeat).encode("ascii")).hexdigest()
    return int(digest[:8], 16), int(digest[8:16], 16)


def trajectory_header(task_id, split, world_id, policy_name, fixed_config, frames, traced,
                      repeat, reset_seed):
    """Every identity a trajectory needs, so the CSV stands on its own.

    The time wording is built from the live env's own step lengths rather than
    written out, so it cannot drift from `n_exec_steps` or `traj_dt`. `repeat`
    and `reset_seed` are in the header because with `--repeats` the world, start,
    goal and policy of five files are identical -- without them the file would
    not say which of the five draws it is.
    """
    def vector(values):
        return json.dumps([float(value) for value in values])
    return [
        "policy-planned and executed trajectory of one endpoint episode",
        "task_id=" + task_id,
        "world=" + world_id,
        "split=" + split,
        "policy=" + policy_name,
        "repeat=" + str(repeat + 1),
        "env_reset_seed=" + str(int(reset_seed)),
        "start=" + vector(fixed_config["start_pos"]),
        "start_yaw=" + repr(float(fixed_config["start_yaw"])),
        "goal=" + vector(fixed_config["goal_pos"]),
        "spawn_offset=" + json.dumps(frames["spawn_offset"]),
        "frame rule: " + frames["rule"],
        "columns x/y/z and yaw are world frame",
        "time base: decision d starts at (d-1)*{} s".format(traced.decision_seconds),
        "time planned: waypoint k is {}*(k+1) s after the base".format(traced.waypoint_seconds),
        "time executed: step k is {}*k s after the base (step 0 repeats the "
        "previous decision's last waypoint)".format(traced.waypoint_seconds),
        "planned: {} waypoints per decision, [pos(3), vel(3), acc(3), yaw, yaw_rate]".format(
            traced.policy_steps),
        "executed: first {} planned waypoints, positions only, one row per time step".format(
            traced.exec_steps),
    ]


def write_task_file(path, task_id, split, world_id, fixed_config, obstacle_source,
                    geometry, frames):
    """`<stem>.task.json`: one episode's identity, world geometry and frames.

    Only written under `--artifacts full`. `geometry` is the obstacle file read
    verbatim, so a pack can be plotted without the source tree; `obstacle_source`
    keeps the path and SHA256 it came from, so the copy can be checked back.
    """
    write_json(path, dict(task_id=task_id, split=split, world_id=world_id,
        fixed_config=fixed_config, obstacle_source=obstacle_source, geometry=geometry,
        frames=frames))


def open_event_stream(path, enabled):
    """The `.trace.jsonl` handle, or a no-op one when the stream is not wanted.

    `--artifacts trajectory` still replays every episode; it just has nowhere to
    put the per-decision events. The adapter then receives `None` and every
    `emit` becomes a no-op (`TraceAdapter.emit`), so the trajectory tables come
    out exactly as in `full`.
    """
    return path.open("x", encoding="utf-8") if enabled else contextlib.nullcontext(None)


def world_root(args, split):
    """Pack-free mode's world root for one split, or a named error.

    Mirrors `evaluate_dppo_uav_formal_endpoint`'s `--worlds-test` /
    `--worlds-layera`. Only the splits the selected cells actually use are
    required, so `--limit 1` on a `rl_test` cell needs no `layerA` root.
    """
    option = {"rl_test": "worlds_test", "layerA": "worlds_layera"}.get(split)
    if option is None or getattr(args, option) is None:
        raise ValueError("pack-free mode needs the {} world root; pass --{}".format(
            split, (option or "worlds_" + split).replace("_", "-")))
    return Path(getattr(args, option)).resolve()


def resolve_inputs(args):
    """The cells to replay and every identity they were resolved from.

    `--source-dir` replays a recorded pack: cell list, config, checkpoint,
    episode banks and obstacles all come from its provenance, every one of them
    is re-bound by SHA256, and `expected` maps each task to the two episodes
    recorded for it. Without it there is no record, so the same inputs are
    resolved from the config and the banks exactly as the formal evaluation
    resolves them and `expected` is None -- the caller then records
    `reproduction_checked=false` rather than inventing a comparison.

    The field order of both cells and obstacle sources follows the bank, so
    `--limit N` makes a smoke run cover the cells a full run starts with.
    """
    from diffusion_policy_3d.rl.dppo_uav.integration import manifest_eval_entries

    if args.source_dir is not None:
        conflicts = [name for name in ("config", "rl_checkpoint", "eval_config_dir", "worlds_test", "worlds_layera")
                     if getattr(args, name) is not None]
        if conflicts:
            raise ValueError("--source-dir fixes these itself; drop " +
                             ", ".join("--" + name.replace("_", "-") for name in conflicts))
        bank, _, provenance = load_evidence(args.source_dir)
        lookup = {bc.task_id: (bc, rl) for bc, rl in bank.pairs}
        if len(lookup) != len(bank.pairs):
            raise ValueError("duplicate task_id in paired bank")
        # Pin original evaluation code. No patch to identity-bearing production files.
        for relative, digest in provenance["evaluation_source_sha256"].items():
            path = (ROOT / relative).resolve()
            if ROOT.resolve() not in path.parents:
                raise ValueError("evaluation source path escapes repository")
            bound_file(path, digest)
        config_path = bound_file(provenance["config_path"], provenance["config_sha256"])
        manifest = bound_file(provenance["source_manifest_path"], provenance["source_manifest_sha256"])
        config_dirs = {bound_file(entry["path"], entry["sha256"]).parent
                       for entry in provenance["fixed_config_files"].values()}
        if len(config_dirs) != 1:
            raise ValueError("fixed episode banks must share their original directory")
        fixed_bank = _load_fixed_bank(config_dirs.pop(), args.benchmark)
        _validate_bank_sources(fixed_bank, manifest_eval_entries(str(manifest)))
        for split, world_id, fixed in fixed_bank:
            bc, _ = lookup[str(fixed.payload["episode_id"])]
            if (bc.config_hash != fixed.config_hash or bc.seed != fixed.seed
                    or (bc.split, bc.world_id) != (split, world_id)):
                raise ValueError("fixed episode content no longer matches original record")
            source = provenance["obstacle_sources"][world_id]
            if source["split"] != split or source["sha256"] != bc.obstacle_hash:
                raise ValueError("obstacle provenance does not match paired record")
            bound_file(source["path"], source["sha256"])
        checkpoint_dir = Path(provenance["checkpoint_dir"])
        if _rl_policy_identity(checkpoint_dir) != bank.pairs[0][1].policy_identity_hash:
            raise ValueError("checkpoint header/integrity identity changed")
        cells = fixed_bank if args.limit is None else fixed_bank[:args.limit]
        obstacle_sources = provenance["obstacle_sources"]
        label = Path(args.source_dir).name
        config_hashes = None
        bc_identity = bank.pairs[0][0].policy_identity_hash
    else:
        missing = [name for name in ("config", "rl_checkpoint", "eval_config_dir")
                   if getattr(args, name) is None]
        if missing:
            raise ValueError("without --source-dir these are required: " +
                             ", ".join("--" + name.replace("_", "-") for name in missing))
        from omegaconf import OmegaConf
        config_path = args.config.resolve()
        checkpoint_dir = args.rl_checkpoint.resolve()
        config_dir = args.eval_config_dir.resolve()
        config = OmegaConf.load(config_path)
        # A template config carries `max_no_ray_hit_fraction: null` until
        # `scripts/calibrate_routea_v17_pointcloud.py` resolves it.  build_system
        # reads the key with a `getattr` default that a present-but-null key
        # never reaches, so without this check the run only fails at `float(None)`
        # -- after the E5 checkpoint has been loaded.  Name the real problem here,
        # in the words the v2 config schema already uses for it.
        if config.environment.max_no_ray_hit_fraction is None:
            raise ValueError("max_no_ray_hit_fraction requires frozen BC calibration")
        manifest_path = Path(config.admission_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (ROOT / manifest_path).resolve()
        fixed_bank = _load_fixed_bank(config_dir, args.benchmark)
        _validate_bank_sources(fixed_bank, manifest_eval_entries(str(manifest_path)))
        cells = fixed_bank if args.limit is None else fixed_bank[:args.limit]
        lookup = None
        bc_identity = None
        # Resolve only the obstacles the selected cells use, so a one-cell smoke
        # run does not need the other split's world root.
        obstacle_sources = {}
        for split, world_id, _ in cells:
            if world_id not in obstacle_sources:
                path = _resolve_obstacle(world_root(args, split), split, world_id)
                obstacle_sources[world_id] = dict(split=split, path=str(path), sha256=sha256(path))
        label = config_path.stem
        manifest = manifest_path
        provenance = None
        config_hashes = {split: sha256(config_dir / "episode_configs_{}.json".format(split))
                         for split in BENCHMARK_SPLITS[args.benchmark]}
    return dict(mode=("replay" if lookup is not None else "collect"), cells=cells,
                total_cells=len(fixed_bank), expected=lookup, bc_identity=bc_identity,
                config_path=config_path, checkpoint_dir=checkpoint_dir,
                manifest_path=manifest, label=label, source_manifest_hash=sha256(manifest),
                config_hashes=config_hashes, obstacle_sources=obstacle_sources,
                provenance=provenance)


def verification(status, checks, selected_cells, total_cells, repeats):
    """`replay_verification.json`: what was run, and what was compared.

    `status` is `matched` when every first repeat reproduced its recorded
    episode, `mismatch` when one did not, and `not_compared` in pack-free mode,
    where there is no recorded episode to compare against. The last case is
    stated rather than left as a missing key, so a reader cannot mistake an
    unreproduced collection for a verified replay.
    """
    return dict(status=status, checks=checks, selected_cells=selected_cells,
                total_cells=total_cells, completed_episodes=len(checks), repeats=repeats,
                comparison=("every first repeat against its recorded episode" if status != "not_compared"
                            else "none: pack-free collection; every repeat is a new draw"))


def run(args):
    write_events = args.artifacts == "full"
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    wanted = tuple(name.strip() for name in args.policies.split(",") if name.strip())
    if not wanted or set(wanted) - {"bc", "rl"}:
        raise ValueError("--policies must be a subset of bc,rl")
    inputs = resolve_inputs(args)
    cells = inputs["cells"]
    lookup = inputs["expected"]
    provenance = inputs["provenance"]
    config_path = inputs["config_path"]
    checkpoint = inputs["checkpoint_dir"]
    output = resolve_output_dir(args.output_dir, "traces_" + args.benchmark, inputs["label"])
    if args.source_dir is not None and output == Path(args.source_dir).resolve():
        raise ValueError("diagnostic output must not replace source endpoint")
    if outside_default_root(output):
        print("warning: output is outside the shared endpoint root {}: {}".format(
            DEFAULT_ROOT, output), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    print("trajectory output: {}".format(output), flush=True)
    write_json(output / "replay_request.json", dict(
        status="prepared_not_completed", source_provenance=provenance,
        input_mode=("recorded_endpoint_pack_replay" if lookup is not None else
                    "pack_free_collection_no_reproduction_check"),
        resolved_inputs=dict(config_path=str(config_path), checkpoint_dir=str(checkpoint),
                             manifest_path=str(inputs["manifest_path"]),
                             manifest_sha256=inputs["source_manifest_hash"],
                             benchmark=args.benchmark, limit=args.limit,
                             obstacle_sources=inputs["obstacle_sources"]),
        selected_cells=len(cells), total_cells=inputs["total_cells"], policies=list(wanted),
        artifacts=args.artifacts,
        repeats=args.repeats,
        reproduction_check=("repeat 1 of every cell against its recorded episode" if lookup is not None
                            else "none: no source pack, so every repeat is a new draw"),
        repeat_seed_rule="repeat 1 replays the bank's recorded seeds and is reproduction-checked; "
                         "repeat n>1 takes sha256('<config_hash>:repeat:<n>') as a fresh env reset "
                         "seed plus an action-seed offset, so it is a new draw rather than a replay",
        numeric_comparison=dict(rtol=1e-6, atol=1e-6),
        scope="fixed_benchmark_trajectory_dump_not_formal_endpoint", latency_measurement_valid=False,
        observer="temporary Python return profiler; original synthesize called once; no RNG draws",
        evaluation_order="fixed_bank order; within each cell repeat 1..N; within each repeat bc then rl",
        per_decision_identity="policy_input.provenance = ObservationProvenance of the input the "
                              "policy consumed for that decision, plus the action seed",
        diagnostic_source_sha256={name: sha256(ROOT / "scripts" / name) for name in
            ("diagnose_dppo_uav_endpoint.py", "endpoint_trajectory_trace_v2.py", "dump_dppo_uav_endpoint_traces.py")}))
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import load_deployment_policy
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    bc_system = build_system(config, ROOT)
    deployment = load_deployment_policy(config, checkpoint, ROOT)
    if provenance is not None and asdict(deployment.checkpoint_header) != provenance["checkpoint_header"]:
        raise ValueError("runtime checkpoint header differs from original provenance")
    if getattr(bc_system.reward, "version", None) != "v17":
        raise ValueError("requires reward v17")
    policies = {
        "bc": _E5Policy(bc_system, use_base_policy=True, identity_hash=bc_system.bc_runtime.identity.content_hash),
        "rl": _E5Policy(deployment.system, use_base_policy=False, identity_hash=_rl_policy_identity(checkpoint)),
    }
    if lookup is not None and policies["bc"].identity_hash != inputs["bc_identity"]:
        raise ValueError("BC policy identity mismatch")
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    checks = []
    stems = set()
    trajectory_stems = set()
    sequence = 0
    for pair_index, (split, world_id, fixed) in enumerate(cells):
        task_id = str(fixed.payload["episode_id"])
        if lookup is not None and task_id not in lookup:
            raise ValueError("fixed bank holds a task outside the paired bank")
        stem = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:20]
        if stem in stems:
            raise ValueError("diagnostic output task hash collision")
        stems.add(stem)
        obstacle = inputs["obstacle_sources"][world_id]
        frames = frame_contract(KinematicUAVEnv._load_spawn_offset(obstacle["path"]))
        task_path = output / (stem + ".task.json")
        if write_events:
            # Copy geometry into the diagnostic pack for self-contained plots.
            geometry = read_json(obstacle["path"])
            write_task_file(task_path, task_id, split, world_id, fixed.payload,
                            obstacle, geometry, frames)
        for repeat in range(args.repeats):
            # Repeat 0 replays the recorded seeds and is the one episode per
            # cell whose record must reproduce; every later repeat is a fresh
            # draw of the env noise and of the policy's own sampling.
            if repeat == 0:
                episode, action_seed_offset = fixed, 0
            else:
                reset_seed, action_seed_offset = repeat_seeds(fixed.config_hash, repeat)
                episode = replace(fixed, seed=reset_seed)
            for order_in_pair, policy_name in enumerate(("bc", "rl")):
                if policy_name not in wanted:
                    continue
                expected = None if lookup is None else lookup[task_id][order_in_pair]
                if expected is None:
                    # No record to reproduce, so the episode carries the identity
                    # of the obstacles it was actually run against.
                    obstacle_hash = obstacle["sha256"]
                    episode_source_hash = (inputs["config_hashes"][split] if split == "layerA"
                                           else inputs["source_manifest_hash"])
                    role = str(fixed.payload["role"])
                else:
                    obstacle_hash = expected.obstacle_hash
                    episode_source_hash = expected.source_manifest_hash
                    role = expected.role
                policy = policies[policy_name]
                adapter = _build_adapter(policy.system, config, world_id, Path(obstacle["path"]), role)
                trace_path = output / "{}.{}.r{}.trace.jsonl".format(stem, policy_name, repeat + 1)
                with open_event_stream(trace_path, write_events) as stream:
                    traced = TraceAdapter(adapter, stream, dict(
                        task_id=task_id, policy=policy_name,
                        pair_index=pair_index, policy_order_in_pair=order_in_pair,
                        repeat=repeat, cell_sequence=sequence,
                        config_hash=fixed.config_hash, policy_identity_hash=policy.identity_hash))
                    sequence += 1
                    try:
                        actual = _evaluate_one(policy_name=policy_name,
                            policy=PolicyRecorder(policy, traced.policy_inputs,
                                                  action_seed_offset=action_seed_offset),
                            adapter=traced, fixed=episode, split=split, world_id=world_id,
                            obstacle_hash=obstacle_hash,
                            source_manifest_hash=episode_source_hash)
                        differences = [] if expected is None else compare_record(expected, actual)
                        traced.emit("episode_end", summary=asdict(actual), reproduction_differences=differences)
                    finally:
                        traced.close()
                trajectory_stem_name = trajectory_stem(
                    world_id, fixed.payload["start_pos"], policy_name, repeat)
                if trajectory_stem_name in trajectory_stems:
                    raise ValueError("two trajectories would share the name " + trajectory_stem_name)
                trajectory_stems.add(trajectory_stem_name)
                trajectory_path = output / (trajectory_stem_name + ".csv")
                write_trajectory_csv(
                    trajectory_path,
                    trajectory_header(task_id, split, world_id, policy_name, fixed.payload, frames,
                                      traced, repeat, episode.seed),
                    traced.planned_rows, traced.executed_rows)
                check = dict(task_id=task_id, policy=policy_name, repeat=repeat,
                             reproduction_checked=(repeat == 0 and expected is not None),
                             trajectory_file=trajectory_path.name,
                             trajectory_sha256=sha256(trajectory_path),
                             planned_rows=len(traced.planned_rows),
                             executed_rows=len(traced.executed_rows),
                             matched=(None if expected is None else not differences),
                             differences=differences,
                             expected=(None if expected is None else asdict(expected)),
                             actual=asdict(actual))
                if write_events:
                    check.update(trace_file=trace_path.name, trace_sha256=sha256(trace_path),
                                 task_sha256=sha256(task_path))
                checks.append(check)
                # Only the reproduction run is held to the recorded episode. A
                # later repeat is expected to differ -- it is a different draw
                # -- so its differences are recorded, not treated as a fault.
                if differences and repeat == 0:
                    write_json(output / "replay_verification.json", verification(
                        "mismatch", checks, len(cells), inputs["total_cells"], args.repeats))
                    print("Replay mismatch; preserved {}, stopped without changing physics or identity.".format(
                        "trace" if write_events else "trajectory and mismatching expected/actual"), flush=True)
                    return 2
        print("traced episodes completed: {}/{} {}".format(
            len(checks) // (len(wanted) * args.repeats), len(cells), task_id), flush=True)
    if len(checks) != len(wanted) * len(cells) * args.repeats:
        raise RuntimeError("selection was not fully replayed")
    write_json(output / "replay_verification.json", verification(
        "matched" if lookup is not None else "not_compared", checks, len(cells),
        inputs["total_cells"], args.repeats))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="recorded endpoint pack to replay; every first repeat is checked "
                             "against the episode recorded there. Omit it to collect from "
                             "--config / --rl-checkpoint / --eval-config-dir instead, in which "
                             "case there is nothing to reproduce and replay_verification.json "
                             "reports not_compared")
    parser.add_argument("--config", type=Path, default=None,
                        help="pack-free mode (no --source-dir): the config the checkpoint was "
                             "trained under; its admission_manifest fixes the evidence source")
    parser.add_argument("--rl-checkpoint", type=Path, default=None,
                        help="pack-free mode: the checkpoint directory to evaluate")
    parser.add_argument("--eval-config-dir", type=Path, default=None,
                        help="pack-free mode: directory holding episode_configs_<split>.json")
    parser.add_argument("--worlds-test", type=Path, default=None,
                        help="pack-free mode: root of the rl_test obstacle files, as passed to "
                             "evaluate_dppo_uav_formal_endpoint.py")
    parser.add_argument("--worlds-layera", type=Path, default=None,
                        help="pack-free mode: root of the layerA obstacle files; only needed "
                             "when the selected cells include a layerA world")
    parser.add_argument("--limit", type=int, default=None,
                        help="replay only the first N cells in bank order; a smoke run then "
                             "covers the cells a full run starts with")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="explicit output directory; omit to get a dated directory "
                             "under the shared endpoint root")
    parser.add_argument("--benchmark", default="fixed32",
                        help="names the output subdirectory traces_<benchmark>")
    parser.add_argument("--policies", default="bc,rl",
                        help="subset of bc,rl to trace; BC is byte-identical across all "
                             "15 cells (verified 0/128 differing on 40/40 fields), so the "
                             "BC traces of one cell serve every cell")
    parser.add_argument("--repeats", type=int, default=1,
                        help="how many times each (world, start, goal) cell is replayed per "
                             "policy; repeat 1 replays the bank's seeds and is reproduction-"
                             "checked, repeats 2..N are fresh draws (see the module docstring)")
    parser.add_argument("--artifacts", choices=("trajectory", "full"), default="trajectory",
                        help="'trajectory' (default) writes only the per-episode trajectory "
                             "CSV plus replay_request/replay_verification; 'full' also writes "
                             "the per-episode .task.json and .trace.jsonl event stream. "
                             "Everything the trajectory needs -- provenance pinning, the "
                             "reproduction check and the plan/executed agreement assertion -- "
                             "runs in both modes; 'full' only adds the per-decision event "
                             "record (reward_metrics, observation hashes, per-frame pointcloud "
                             "seeds). The two consumers that need that stream, "
                             "render_dppo_endpoint_traces.py and "
                             "compare_endpoint_decision_identity.py, have no input in "
                             "'trajectory' mode.")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
