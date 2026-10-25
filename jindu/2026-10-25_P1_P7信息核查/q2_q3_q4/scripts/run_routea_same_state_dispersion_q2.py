#!/usr/bin/env python3
"""Q2: same-state behavior dispersion, deterministic vs random sampling.

18 frozen states x 20 draws x {BC, u128, u256} x {deterministic True, False},
open-loop only. Initial noise is paired across policies AND modes via
paired_seed(state_id, draw) (same seed -> same initial x_T by construction).

Reuses the accepted recovery machinery (read_bank / replay_state /
predict_with_noise / paired_seed). No tracked source file is modified: the
sampling-mode flag is flipped by a local subclass of _E5Policy, so the
formal-endpoint identity closure is untouched and checkpoint loading keeps
its normal identity verification.
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

W2560 = ROOT / "data/outputs/dppo_uav_rebuild/routea_common_b2560_seed202609203_v1"
REF = ROOT / "data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02"
BANK_PATH = ROOT / "scripts/routea_paired_support/a1_mechanism_bank_v1.json"
DEFAULT_OUT = Path("/home/cjj/AUV/program/results/routea_paired_evaluation/"
                   "routea_q2_samestate_dispersion_20261025_run01")
MODES = {"deterministic": True, "random": False}


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class _Q2PolicyMixin:
    """Same contract as _E5Policy, with a sampling-mode switch.

    The body mirrors _E5Policy.predict exactly except
    deterministic=self.deterministic; the formal endpoint source is not
    modified (identity closure stays byte-identical).
    """

    deterministic = True

    def predict(self, observation, observation_context, *, action_seed):
        import torch

        torch.manual_seed(int(action_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(action_seed))
        with torch.no_grad():
            encoded = self.system.observation_adapter.encode(
                observation,
                observation_context["pointcloud_meta_history"],
                episode_identity_hash=observation_context["episode_identity_hash"],
                decision_index=observation_context["decision_index"],
            )
            self.system.observation_adapter.verify(encoded)
            self.last_provenance = encoded.provenance
            condition = encoded.condition
            samples = self.system.official_runtime(
                cond=condition,
                deterministic=self.deterministic,
                return_chain=True,
                use_base_policy=self.use_base_policy,
            )
            full = self.system.bc_runtime.policy.normalizer["action"].unnormalize(
                samples.trajectories
            )
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
            ActionBundle,
        )

        return ActionBundle.from_full(full)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES),
                        default=sorted(MODES))
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if sys.version_info[:3] != (3, 8, 20):
        raise RuntimeError("run in the fixed idp3 Python 3.8.20 environment")
    if out.exists() and any(out.iterdir()) and not (out / "request.json").exists():
        raise FileExistsError("output directory must be new or empty: " + str(out))
    out.mkdir(parents=True, exist_ok=True)

    import fcntl
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with torch.jit.optimized_execution(False):
        with (out / ".q2_controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _run(ROOT, out, [m for m in args.modes])


def _run(root, out, modes):
    import numpy as np
    from scripts.routea_paired_support.reference import check_reference
    from scripts.routea_paired_support.runtime import (
        common_system,
        load_actor,
        resolve_catalogue,
    )
    from scripts.routea_paired_support.recovery import (
        paired_seed,
        predict_with_noise,
        read_bank,
        replay_state,
        verify_pairing,
    )
    from scripts.routea_paired_support.core import write_json
    from scripts.routea_paired_support.workflow import policy_order
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy

    class _Q2Policy(_Q2PolicyMixin, _E5Policy):
        pass

    bank = read_bank(BANK_PATH)
    task_bank = check_reference(root, REF, allow_space_expansion=True)
    checkpoints = [str(W2560 / "checkpoint_update_000128"),
                   str(W2560 / "checkpoint_update_000256")]
    cat = resolve_catalogue(root, checkpoints)
    if not (out / "request.json").exists():
        write_json(out / "request.json", {
            "bank": str(BANK_PATH), "bank_hash": bank["content_hash"],
            "reference": str(REF), "checkpoints": checkpoints,
            "modes": {m: MODES[m] for m in modes},
            "paired_noise": "paired_seed(state_id, draw) identical across policies and modes",
        })
    if not (out / "models.json").exists():
        write_json(out / "models.json", cat)

    system, config = common_system(root, cat)
    snapshots = {}
    started = time.monotonic()
    print("REPLAY_STATES_START", flush=True)
    for state in bank["states"]:
        snapshots[state["id"]] = replay_state(system, config, task_bank, state)
    print("REPLAY_STATES_DONE", len(snapshots), "states", flush=True)

    summary_rows = []
    for mode_name in modes:
        deterministic = MODES[mode_name]
        mode_dir = out / mode_name
        for name in policy_order(cat):
            descriptor = cat["policies"][name]
            if name != "bc":
                load_actor(root, system, descriptor)
                anchor = bank["states"][0]
                _, obs, ctx = snapshots[anchor["id"]]
                # The guard must resample in the SAME mode as the saved bc
                # samples; the original _E5Policy hardcodes deterministic=True.
                base = _Q2Policy(system, use_base_policy=True,
                                 identity_hash=cat["common"]["bc_identity_hash"])
                base.deterministic = deterministic
                bundle, noise = predict_with_noise(base, obs, ctx,
                                                   paired_seed(anchor["id"], 0))
                saved = _read_json(mode_dir / "bc" / (anchor["id"] + ".json"))
                actual = np.asarray(bundle.full.detach().cpu(), np.float32).reshape(16, 11)
                if (not np.array_equal(actual,
                                       np.asarray(saved["samples"][0]["full_action_16x11"], np.float32))
                        or noise != saved["samples"][0]["initial_noise_hash"]):
                    raise ValueError("BC branch changed after loading " + name)
                check_path = mode_dir / name / "bc_disabled_branch_check.json"
                if not check_path.exists():
                    write_json(check_path,
                               dict(status="pass", actual_use_base_policy=True,
                                    state_sha256=descriptor["state_sha256"]))
            identity = (cat["common"]["bc_identity_hash"] if name == "bc"
                        else descriptor["runtime_identity_hash"])
            policy = _Q2Policy(system, use_base_policy=descriptor["use_base_policy"],
                               identity_hash=identity)
            policy.deterministic = deterministic
            for state in bank["states"]:
                state_dir = mode_dir / name
                state_dir.mkdir(parents=True, exist_ok=True)
                state_path = state_dir / (state["id"] + ".json")
                if state_path.exists():
                    # resume: the pairing and bc-branch guards still run below
                    result = _read_json(state_path)
                    summary_rows.append(dict(policy=name, mode=mode_name,
                                             state_id=state["id"],
                                             group=state["group"],
                                             n_draws=len(result["samples"])))
                    print("SKIP", mode_name, name, state["id"], flush=True)
                    continue
                adapter, obs, ctx = snapshots[state["id"]]
                pre = state["pre_state"]
                samples = []
                for draw in range(bank["draws"]):
                    seed = paired_seed(state["id"], draw)
                    bundle, noise_hash = predict_with_noise(policy, obs, ctx, seed)
                    full = np.asarray(bundle.full.detach().cpu(), np.float32)
                    full = full.reshape(16, 11)
                    h4_pos = (np.asarray(pre["position"], np.float64)
                              + full[0:4, 0:3].sum(axis=0).astype(np.float64)).tolist()
                    h8_pos = (np.asarray(pre["position"], np.float64)
                              + full[0:8, 0:3].sum(axis=0).astype(np.float64)).tolist()
                    h4_yaw = float(_wrap(pre["yaw"] + full[0:4, 9].sum()))
                    h8_yaw = float(_wrap(pre["yaw"] + full[0:8, 9].sum()))
                    samples.append(dict(
                        draw=draw, seed=seed, initial_noise_hash=noise_hash,
                        h4_pos=h4_pos, h8_pos=h8_pos,
                        h4_yaw=h4_yaw, h8_yaw=h8_yaw,
                        full_action_16x11=full.tolist(),
                        policy_input_hash=policy.last_provenance.policy_input_hash))
                write_json(state_dir / (state["id"] + ".json"), dict(
                    policy=name, mode=mode_name, deterministic=deterministic,
                    policy_identity=identity, state_id=state["id"],
                    group=state["group"], samples=samples))
                summary_rows.append(dict(policy=name, mode=mode_name,
                                         state_id=state["id"], group=state["group"],
                                         n_draws=len(samples)))
                print("SAMPLE", mode_name, name, state["id"],
                      "seconds", round(time.monotonic() - started, 1), flush=True)
            if name != "bc":
                for state in bank["states"]:
                    verify_pairing(
                        _read_json(mode_dir / "bc" / (state["id"] + ".json"))["samples"],
                        _read_json(mode_dir / name / (state["id"] + ".json"))["samples"])
    # cross-mode pairing: same (state, draw) -> same initial noise hash
    cross = {}
    for mode_name in modes:
        for state in bank["states"]:
            for sample in _read_json(out / mode_name / "bc" /
                                     (state["id"] + ".json"))["samples"]:
                key = (state["id"], sample["draw"])
                if key in cross and cross[key] != sample["initial_noise_hash"]:
                    raise ValueError("cross-mode initial noise mismatch: " + str(key))
                cross[key] = sample["initial_noise_hash"]
    write_json(out / "run_complete_001.json", dict(
        status="complete", bank_hash=bank["content_hash"],
        modes=modes, policies=list(policy_order(cat)),
        n_states=len(bank["states"]), draws=bank["draws"],
        summary_rows=summary_rows,
        cross_mode_paired_noise_keys=len(cross),
        seconds=round(time.monotonic() - started, 1)))
    print("Q2_SAMPLING_COMPLETE", str(out), flush=True)


def _read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
