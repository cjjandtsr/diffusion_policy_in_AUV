#!/usr/bin/env python3
"""Localise a BC/RL divergence to the evaluation pipeline or to the policy.

Reads the traces written by `dump_dppo_uav_endpoint_traces.py` and, for every
paired episode, walks the two policies' decision steps side by side.

The discriminator is ordering, not magnitude. Both policies share an initial
condition and are deterministic given their input, so:

  * while the physical state is identical, the observation hashes and the
    policy-input hash must also be identical; a difference there means the
    pipeline handed the two policies different inputs and the divergence is
    upstream of the policy;
  * if every input identity matches and `full_action` differs, the split comes
    from the weights, which is the only place a policy difference can live.

Each pair is placed in exactly one class, by the earliest divergence:

  identical           no divergence on any compared field, equal step counts
  truncation_only     identical for as long as both ran; one simply ran longer
  order_suspect       episode identity or action seed differs at step 1
  pipeline_suspect    an input hash differs while the physical state has not
  policy_divergence   inputs identical up to the first action difference
  state_carried       inputs differ only after the states already diverged
  incomplete          a trace is missing, empty, or has no episode_end

The two episodes commonly run different numbers of decisions -- BC may arrive
in 28 steps where RL runs to the 60-step limit. That is the outcome difference
itself, so it is reported as `step_count_delta` alongside the class rather than
being a class: the shared prefix still carries the whole answer, and reading a
step-count difference as "unclassifiable" would discard it.

`state_carried` is not a defect: once the two policies take different actions
their states legitimately differ and every later input differs with them. It is
only the FIRST divergence that carries information.

This reads traces only. It never re-runs the evaluator.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

CLASSES = (
    "identical",
    "truncation_only",
    "order_suspect",
    "pipeline_suspect",
    "policy_divergence",
    "state_carried",
    "incomplete",
)


def read_jsonl(path):
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def transitions(events):
    return [event for event in events if event.get("event") == "transition"]


def _array(value):
    return None if value is None else np.asarray(value, dtype=np.float64)


def _positions_equal(left, right, atol):
    a, b = _array(left), _array(right)
    if a is None or b is None:
        return a is b
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=atol))


def _actions_equal(left, right, atol):
    a, b = _array(left), _array(right)
    if a is None or b is None:
        return a is b
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=0.0, atol=atol))


def _action_max_delta(left, right):
    a, b = _array(left), _array(right)
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.max(np.abs(a - b)))


def _provenance(transition, name):
    block = transition.get("policy_input") or {}
    provenance = block.get("provenance") or {}
    return provenance.get(name)


def compare_pair(bc_events, rl_events, atol):
    """Walk two episodes step by step; return the class and the first splits."""
    bc_steps, rl_steps = transitions(bc_events), transitions(rl_events)
    result = dict(
        bc_decisions=len(bc_steps),
        rl_decisions=len(rl_steps),
        step_count_delta=len(rl_steps) - len(bc_steps),
        compared_decisions=min(len(bc_steps), len(rl_steps)) if bc_steps and rl_steps else 0,
        first_state_divergence=None,
        first_observation_divergence=None,
        first_input_divergence=None,
        first_action_divergence=None,
        action_max_delta=None,
        detail=None,
    )
    if not bc_steps or not rl_steps:
        return "incomplete", result
    shared = min(len(bc_steps), len(rl_steps))
    for index in range(shared):
        bc, rl = bc_steps[index], rl_steps[index]
        step = bc.get("decision_step")
        if step != rl.get("decision_step"):
            return "incomplete", dict(result, detail="decision_step misaligned at index {}".format(index))

        if result["first_state_divergence"] is None and not _positions_equal(
            (bc.get("before") or {}).get("position"), (rl.get("before") or {}).get("position"), atol
        ):
            result["first_state_divergence"] = step

        bc_obs = (bc.get("observation_before") or {}).get("pointcloud_sha256")
        rl_obs = (rl.get("observation_before") or {}).get("pointcloud_sha256")
        if result["first_observation_divergence"] is None and bc_obs != rl_obs:
            result["first_observation_divergence"] = step

        bc_input = _provenance(bc, "policy_input_hash")
        rl_input = _provenance(rl, "policy_input_hash")
        if result["first_input_divergence"] is None and bc_input != rl_input:
            result["first_input_divergence"] = step
            result["detail"] = dict(
                step=step,
                bc_policy_input_hash=bc_input,
                rl_policy_input_hash=rl_input,
                bc_raw_frame_hash=_provenance(bc, "raw_frame_hash"),
                rl_raw_frame_hash=_provenance(rl, "raw_frame_hash"),
                bc_episode_identity_hash=_provenance(bc, "episode_identity_hash"),
                rl_episode_identity_hash=_provenance(rl, "episode_identity_hash"),
                state_identical=result["first_state_divergence"] is None,
            )

        if result["first_action_divergence"] is None and not _actions_equal(
            bc.get("full_action"), rl.get("full_action"), atol
        ):
            result["first_action_divergence"] = step
            result["action_max_delta"] = _action_max_delta(bc.get("full_action"), rl.get("full_action"))
            # The inherited RNG state legitimately differs between the two
            # policies (they sit at different points in the cell's sequence), so
            # this is recorded as context, never as a pass/fail criterion. It
            # matters only if the action split has no input difference to
            # explain it.
            result["detail"] = dict(result["detail"] or {}, first_action_step=step,
                bc_inherited_rng=(bc.get("policy_input") or {}).get("inherited_rng_state_hash"),
                rl_inherited_rng=(rl.get("policy_input") or {}).get("inherited_rng_state_hash"))

    # The action seed is a pure function of (config_hash, decision_step), both of
    # which are shared inside a pair, so any difference means the run itself was
    # driven out of order -- a different fault from an input difference.
    for index in range(shared):
        bc_seed = (bc_steps[index].get("policy_input") or {}).get("action_seed")
        rl_seed = (rl_steps[index].get("policy_input") or {}).get("action_seed")
        if bc_seed != rl_seed:
            result["action_seed_divergence"] = bc_steps[index].get("decision_step")
            break
    else:
        result["action_seed_divergence"] = None
    if result["action_seed_divergence"] is not None:
        return "order_suspect", result

    state = result["first_state_divergence"]
    observation = result["first_observation_divergence"]
    input_step = result["first_input_divergence"]
    action = result["first_action_divergence"]

    if observation is None and input_step is None and action is None:
        # Same behaviour for as long as both ran; the only difference is that one
        # kept going. Worth separating from `identical`: it says the policies did
        # not diverge, they just stopped at different points.
        if result["step_count_delta"]:
            return "truncation_only", result
        return "identical", result

    # An input split at a step where the physics is still shared cannot be the
    # policy's doing: the policy is a function of the input it was handed.
    if observation is not None or input_step is not None:
        earliest_input = min(step for step in (observation, input_step) if step is not None)
        if state is None or earliest_input < state:
            return "pipeline_suspect", result
        return "state_carried", result

    # Every input matched, yet the action split -- that is the weights.
    return "policy_divergence", result


def compare_cell(traces_dir, atol):
    traces_dir = Path(traces_dir)
    pairs = defaultdict(dict)
    tasks = {}
    for path in sorted(traces_dir.glob("*.trace.jsonl")):
        stem, _, tail = path.name.partition(".")
        policy = tail.split(".")[0]
        events = read_jsonl(path)
        header = events[0] if events else {}
        pairs[header.get("task_id", stem)][policy] = events
        task_file = traces_dir / (stem + ".task.json")
        if task_file.exists():
            tasks[header.get("task_id", stem)] = json.loads(task_file.read_text(encoding="utf-8"))
    records = []
    for task_id, sides in pairs.items():
        task = tasks.get(task_id, {})
        payload = task.get("fixed_config") or {}
        if set(sides) != {"bc", "rl"}:
            records.append(dict(task_id=task_id, world_id=task.get("world_id"), split=task.get("split"),
                                role=payload.get("role"), verdict="incomplete", detail="missing policy side"))
            continue
        verdict, result = compare_pair(sides["bc"], sides["rl"], atol)
        records.append(dict(task_id=task_id, world_id=task.get("world_id"), split=task.get("split"),
                            role=payload.get("role"), verdict=verdict, **result))
    return records


def summarise(records):
    totals = Counter(record["verdict"] for record in records)
    by_role = defaultdict(Counter)
    for record in records:
        by_role[str(record.get("role"))][record["verdict"]] += 1
    return totals, by_role


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True,
                        help="one cell trace dir, or a parent holding several")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-9,
                        help="numeric tolerance for position/action comparison")
    args = parser.parse_args(argv)

    if list(args.traces.glob("*.trace.jsonl")):
        cells = [args.traces]
    else:
        cells = sorted(path for path in args.traces.iterdir() if path.is_dir())
    all_records = []
    for cell in cells:
        records = compare_cell(cell, args.atol)
        for record in records:
            record["cell"] = cell.name
        all_records.extend(records)
        totals, by_role = summarise(records)
        print("[{}] {} pairs".format(cell.name, len(records)))
        for name in CLASSES:
            if totals.get(name):
                print("    {:20s} {}".format(name, totals[name]))
        for role, counter in sorted(by_role.items()):
            print("    role {:24s} {}".format(role, dict(counter)))

    if args.output:
        args.output.write_text(json.dumps(dict(records=all_records), indent=2, sort_keys=True),
                               encoding="utf-8")
        print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
