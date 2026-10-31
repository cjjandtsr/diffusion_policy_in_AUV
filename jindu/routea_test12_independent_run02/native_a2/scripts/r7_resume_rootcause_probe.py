"""r7_resume_rootcause_probe.py — F4 closure guide §3: regression-grade
probe for the same-seed resume divergence.

Replaces the one-shot probe 9 (/tmp/probe9_step_dump.py) whose defects
are now contract-enforced here:

  * hooks wrap step_at (the training path) — never step (the eval path);
  * capture_update records ONLY the requested update (the old probe
    recorded updates 1..3 concatenated and needed the offset-2213 tail
    slice to align);
  * the rollout frame reads the REAL DPPORolloutSample fields
    (chains_policy / old_logprobs / executed_action) and hard-fails on
    a missing field — never records None (§3.1 #3);
  * observation components split: obs_pc_hash / obs_agent_hash /
    obs_combined_hash, so a first diff reports WHICH component changed;
  * compare_dumps never zip-truncates: unequal lengths FAIL explicitly
    with both lengths (§3.1 #5).

Recording schema (guide §3.2):

  per rollout frame:
    absolute_update / relative_transition_index / active_env_ids /
    episode_meta {env_id: episode_id, task_uid, episode_seed,
    episode_step} / python / numpy / torch_cpu / torch_cuda global RNG
    hashes / owned Generator hashes / obs_pc_hash / obs_agent_hash /
    obs_combined_hash / chains_policy_step0_hash / chains_policy_hash /
    old_logprobs_hash / executed_action_hash

  per step_at:
    slot / done / pos_hash / yaw_hash / vel_hash / episode_step /
    disturb_idx / count_stuck / prev_dist / episode_rng_seed /
    slot_rng_hash / pc_history_hash / state_history_hash / reward /
    terminated / truncated / outcome

Raw kinematic values (pos/yaw/vel) are recorded only inside the raw
window — configured via raw_window=(lo, hi), or auto-derived as
[first_diff-2, first_diff+2] when a reference dump is attached and
stop_after_first_diff_context is set (§3.3).

Mechanics (§3.3): incremental .tmp -> fsync -> atomic rename every
flush_every records; nonzero exit on hook miss, missing field, layer
hash mismatch, or a half-written artifact.

The required env surface is the REAL kinematic env's internals
(code-verified on the real adapter by probe 9): _drone_pos, _drone_yaw,
_drone_vel, episode_step, _disturb_idx, _count_stuck, _prev_dist,
_episode_rng_seed, _pc_history, _state_history, _rng. A mock without
that surface fails closed (ProbeContractError) — never a silent skip.
outcome is an optional enrichment (recorded when slot_record provides
it); every other field must exist.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from diffusion_policy_3d.rl.dppo_uav.train_loop import _canonical_bytes  # noqa: E402

SCHEMA_VERSION = "r7-resume-probe-v1"

# Guide §3.2 step schema — required env internals per slot, in compare
# order. Anything else missing is recorded as a hard failure.
REQUIRED_ENV_ATTRS = (
    "_drone_pos", "_drone_yaw", "_drone_vel", "episode_step",
    "_disturb_idx", "_count_stuck", "_prev_dist", "_episode_rng_seed",
    "_pc_history", "_state_history", "_rng",
)

FRAME_ORDER = (
    "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
    "owned_generators",
    "obs_pc_hash", "obs_agent_hash", "obs_combined_hash",
    "chains_policy_step0_hash", "chains_policy_hash",
    "old_logprobs_hash", "executed_action_hash",
)

STEP_ORDER = (
    "done", "pos_hash", "yaw_hash", "vel_hash",
    "episode_step", "disturb_idx", "count_stuck", "prev_dist",
    "episode_rng_seed", "slot_rng_hash",
    "pc_history_hash", "state_history_hash",
    "reward", "terminated", "truncated", "outcome",
)

# The REAL sample_dppo_rollout result fields (phase1a_actor_adapter.py
# DPPORolloutSample) — the probe must read exactly these; res.chain does
# NOT exist.
ROLLOUT_RESULT_FIELDS = ("chains_policy", "old_logprobs", "executed_action")


class ProbeContractError(RuntimeError):
    """A probe contract violation — hook miss, missing field, length
    mismatch, or a half-written artifact. Maps to a nonzero exit."""


def _h(*objs) -> str:
    h = hashlib.sha256()
    for o in objs:
        h.update(_canonical_bytes(o))
    return h.hexdigest()


def _rng_hashes() -> Dict[str, Optional[str]]:
    out = {
        "python_rng": _h(random.getstate()),
        "numpy_rng": _h(np.random.get_state()),
        "torch_cpu_rng": _h(torch.get_rng_state()),
        "torch_cuda_rng": None,
    }
    if torch.cuda.is_available():
        out["torch_cuda_rng"] = _h(torch.cuda.get_rng_state())
    return out


def _owned_generator_hashes(policy) -> List[str]:
    """All torch.Generator instances the policy object graph owns — the
    streams sample_dppo_rollout could consume beyond the global CUDA
    RNG (the x_T noise draw). Empty list when the policy owns none."""
    out: List[str] = []
    for obj in (policy, getattr(policy, "model", None)):
        if obj is None:
            continue
        for value in vars(obj).values():
            if isinstance(value, torch.Generator):
                out.append(_h(value.get_state()))
    return out


def _slot_rng_state(e) -> Any:
    rng = getattr(e, "_rng", None)
    if rng is None:
        return None
    bit = getattr(rng, "bit_generator", None)
    if bit is not None:
        return bit.state  # numpy>=2: Generator.__getstate__() is None
    return rng.get_state()


def _native(value):
    """JSON-native scalar conversion (numpy scalars / torch 0-d)."""
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return value


def _slot_env(adapter, env_id: int):
    envs = getattr(adapter, "_envs", None)
    if envs is not None:
        return envs[env_id]
    return adapter


@dataclass(frozen=True)
class ProbeConfig:
    out_path: Path
    capture_update: Optional[int] = None    # record only from this update
    expect_layer3: Optional[str] = None     # finalize hard-fails on mismatch
    raw_window: Optional[Tuple[int, int]] = None  # (lo, hi) relative steps
    reference: Optional[Dict[str, list]] = None   # other side's frames/steps
    stop_after_first_diff_context: Optional[int] = None
    flush_every: int = 20


class RootCauseProbe:
    """Guide §3.2/3.3 recorder — attach to the training path, drive with
    begin_update, finish with finalize (nonzero exit on contract
    failure)."""

    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.frames: List[dict] = []
        self.steps: List[dict] = []
        self._update = 0
        self._frame_idx = 0
        self._step_idx = 0
        self._armed = cfg.capture_update is None
        self._env = None
        self._env_attached = False
        self._rollout_installed = False
        self._rollout_wrapper = None
        self._rollout_owner = None
        self._rollout_name = None
        self._rollout_orig = None
        self._rollout_hits = 0
        self._step_installed = False
        self._step_wrapper = None
        self._step_cls = None
        self._step_orig = None
        self._step_hits = 0
        self._failures: List[str] = []
        self._since_flush = 0
        self._first_diff: Optional[dict] = None
        self._auto_raw: Optional[Tuple[int, int]] = None
        self._stopped = False

    # ── public counts (test-visible) ────────────────────────────────
    @property
    def rollout_records(self) -> int:
        return len(self.frames)

    @property
    def step_records(self) -> int:
        return len(self.steps)

    # ── hook installation (fail-closed: a silent no-op is a failure) ─
    def install_rollout_hook(self, owner, name: str = "sample_dppo_rollout"):
        orig = getattr(owner, name, None)
        if orig is None:
            raise ProbeContractError(
                f"{getattr(owner, '__name__', owner)!r} has no {name} — "
                "hook would be a silent no-op")
        self._rollout_orig = orig
        self._rollout_owner = owner
        self._rollout_name = name
        probe = self

        def wrapped(policy, obs_b, contract, **kw):
            res = orig(policy, obs_b, contract, **kw)
            probe.record_rollout(policy, obs_b, res)
            return res

        setattr(owner, name, wrapped)
        self._rollout_wrapper = wrapped
        self._rollout_installed = True
        if getattr(owner, name) is not wrapped:
            raise ProbeContractError("rollout hook did not resolve on owner")

    def install_step_hook(self, cls):
        orig = getattr(cls, "step_at", None)
        if orig is None:
            raise ProbeContractError(
                f"{cls.__name__} has no step_at — hook would be a silent "
                "no-op")
        self._step_orig = orig
        self._step_cls = cls
        probe = self

        def wrapped(adapter, env_ids, executed_actions, **kw):
            res = orig(adapter, env_ids, executed_actions, **kw)
            probe.record_step(adapter, env_ids, res)
            return res

        setattr(cls, "step_at", wrapped)
        self._step_wrapper = wrapped
        self._step_installed = True

    def attach_env(self, env) -> None:
        """Bind the env/adapter. Verifies the step hook actually resolved
        on the object the training loop will call — the probe-9
        "silent no-op burns 40 minutes" failure mode."""
        self._env = env
        self._env_attached = True
        if self._step_installed and type(env).step_at is not self._step_wrapper:
            raise ProbeContractError(
                f"step hook did not resolve on {type(env).__name__} — "
                "records would be empty")

    def uninstall(self) -> None:
        if self._rollout_installed and self._rollout_owner is not None:
            setattr(self._rollout_owner, self._rollout_name,
                    self._rollout_orig)
            self._rollout_installed = False
        if self._step_installed and self._step_cls is not None:
            setattr(self._step_cls, "step_at", self._step_orig)
            self._step_installed = False

    # ── update lifecycle ────────────────────────────────────────────
    def begin_update(self, n: int) -> None:
        self._update = n
        if self.cfg.capture_update is None:
            return
        if n == self.cfg.capture_update:
            # Never a 1..N concatenation: the capture boundary CLEARS
            # earlier frames (probe-9 defect) and resets the relative
            # transition counter.
            self.frames.clear()
            self.steps.clear()
            self._frame_idx = 0
            self._step_idx = 0
            self._armed = True
        elif n > self.cfg.capture_update:
            self._armed = True

    # ── recording ───────────────────────────────────────────────────
    def record_rollout(self, policy, obs_b: Dict[str, Any], res) -> None:
        self._rollout_hits += 1
        if not self._armed or self._stopped:
            return
        if not self._env_attached:
            raise ProbeContractError(
                "attach_env() must be called before record_rollout")
        for field in ROLLOUT_RESULT_FIELDS:
            if getattr(res, field, None) is None:
                raise ProbeContractError(
                    f"rollout result missing field '{field}' — never "
                    "record None")
        for key in ("point_cloud", "agent_pos"):
            if obs_b.get(key) is None:
                raise ProbeContractError(
                    f"obs_b missing component '{key}' — component split "
                    "is the probe contract")
        n_envs = int(getattr(self._env, "n_envs", 1))
        active = [i for i in range(n_envs) if self._env.is_active(i)]
        meta = {}
        for env_id in active:
            e = _slot_env(self._env, env_id)
            rec = self._env.slot_record(env_id)
            spec = self._env.slot_spec(env_id)
            meta[str(env_id)] = {
                "episode_id": getattr(rec, "episode_id", None),
                "task_uid": getattr(spec, "world_uid", None),
                "episode_seed": getattr(rec, "episode_seed", None),
                "episode_step": int(getattr(e, "episode_step", -1)),
            }
        chains = res.chains_policy
        frame = {
            "absolute_update": self._update,
            "relative_transition_index": self._frame_idx,
            "active_env_ids": active,
            "episode_meta": meta,
            **_rng_hashes(),
            "owned_generators": _owned_generator_hashes(policy),
            "obs_pc_hash": _h(obs_b["point_cloud"]),
            "obs_agent_hash": _h(obs_b["agent_pos"]),
            "obs_combined_hash": _h(obs_b["point_cloud"], obs_b["agent_pos"]),
            "chains_policy_step0_hash": _h(chains[:, 0]),
            "chains_policy_hash": _h(chains),
            "old_logprobs_hash": _h(res.old_logprobs),
            "executed_action_hash": _h(res.executed_action),
        }
        self.frames.append(frame)
        self._frame_idx += 1
        self._maybe_online_compare()
        self._maybe_flush()

    def record_step(self, adapter, env_ids, res) -> None:
        self._step_hits += 1
        if not self._armed or self._stopped:
            return
        for i, env_id in enumerate([int(v) for v in env_ids]):
            e = _slot_env(adapter, env_id)
            missing = [a for a in REQUIRED_ENV_ATTRS
                       if not hasattr(e, a)]
            if missing:
                raise ProbeContractError(
                    f"slot {env_id} env missing required internals: "
                    f"{missing} — probe records FAIL closed, never "
                    "partially")
            # is_active/slot_record live on the ADAPTER, not the inner
            # env (environment_adapter.py:1040/1217/1221; the inner
            # KinematicUAVEnv has neither).
            rec = (adapter.slot_record(env_id)
                   if hasattr(adapter, "slot_record") else None)
            step = {
                "slot": env_id,
                "done": bool(not adapter.is_active(env_id)),
                "pos_hash": _h(e._drone_pos),
                "yaw_hash": _h(e._drone_yaw),
                "vel_hash": _h(e._drone_vel),
                "episode_step": int(e.episode_step),
                "disturb_idx": int(e._disturb_idx),
                "count_stuck": int(e._count_stuck),
                "prev_dist": (None if e._prev_dist is None
                              else float(e._prev_dist)),
                "episode_rng_seed": (None if e._episode_rng_seed is None
                                     else int(e._episode_rng_seed)),
                "slot_rng_hash": _h(_slot_rng_state(e)),
                "pc_history_hash": _h(e._pc_history),
                "state_history_hash": _h(e._state_history),
                "reward": _native(res.reward[i]),
                "terminated": _native(res.terminated[i]),
                "truncated": _native(res.truncated[i]),
                "outcome": (getattr(getattr(rec, "outcome", None), "name",
                                    getattr(rec, "outcome", None))
                            if rec is not None else None),
            }
            if self._in_raw_window(self._step_idx):
                step["pos"] = [float(v) for v in np.asarray(
                    e._drone_pos).reshape(-1)]
                step["yaw"] = float(e._drone_yaw)
                step["vel"] = [float(v) for v in np.asarray(
                    e._drone_vel).reshape(-1)]
            self.steps.append(step)
            self._step_idx += 1
        self._maybe_flush()

    # ── online compare / early stop (§3.3) ──────────────────────────
    def _maybe_online_compare(self) -> None:
        if self.cfg.reference is None:
            return
        if self._first_diff is None:
            out = compare_dumps(
                {"frames": self.frames, "steps": self.steps},
                self.cfg.reference,
            )
            if out["first_diff"] is not None:
                self._first_diff = out["first_diff"]
                lo = max(0, self._first_diff["index"] - 2)
                hi = self._first_diff["index"] + 2
                self._auto_raw = (lo, hi)
        if (self._first_diff is not None
                and self.cfg.stop_after_first_diff_context is not None):
            limit = self._first_diff["index"] + \
                self.cfg.stop_after_first_diff_context
            if self._frame_idx - 1 >= limit:
                self._stopped = True

    def _in_raw_window(self, index: int) -> bool:
        lo, hi = self._auto_raw or self.cfg.raw_window or (None, None)
        if lo is None:
            return False
        return lo <= index <= hi

    # ── persistence (§3.3: .tmp -> fsync -> atomic rename) ──────────
    def _artifact(self, layer3: Optional[str]) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "capture_update": self.cfg.capture_update,
            "first_diff": self._first_diff,
            "stopped_after_first_diff": self._stopped,
            "layer3": layer3,
            "expected_layer3": self.cfg.expect_layer3,
            "hook_hits": {"rollout": self._rollout_hits,
                          "step": self._step_hits},
            "failures": self._failures,
            "frames": self.frames,
            "steps": self.steps,
        }

    def _write_atomic(self, layer3: Optional[str]) -> None:
        path = Path(self.cfg.out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._artifact(layer3), f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def _maybe_flush(self) -> None:
        self._since_flush += 1
        if self._since_flush >= self.cfg.flush_every:
            self.flush()

    def flush(self) -> None:
        if self.cfg.out_path is not None:
            self._write_atomic(None)
            self._since_flush = 0

    def finalize(self, layer3: Optional[str] = None) -> int:
        """Write the artifact and return the process exit code: 0 on a
        clean capture, 1 on any contract failure (hook miss, missing
        fields, layer-3 mismatch)."""
        if self._armed and not self._env_attached:
            self._failures.append("attach_env never called")
        if self._rollout_installed and self._rollout_hits == 0:
            self._failures.append("rollout hook never hit")
        if self._step_installed and self._step_hits == 0:
            self._failures.append("step hook never hit")
        if self.cfg.expect_layer3 is not None:
            if layer3 is None:
                self._failures.append(
                    "expect_layer3 set but finalize(layer3=None)")
            elif layer3 != self.cfg.expect_layer3:
                self._failures.append(
                    f"layer3 mismatch: got {layer3[:16]}… expected "
                    f"{self.cfg.expect_layer3[:16]}…")
        self._write_atomic(layer3)
        return 1 if self._failures else 0


# ═══════════════════════════════════════════════════════════════════
# comparison — never a silent zip-truncate (§3.1 #5)
# ═══════════════════════════════════════════════════════════════════

def compare_dumps(control: dict, resume: dict, *, offset: int = 0) -> dict:
    """Component-level first-diff between two probe artifacts.

    ``offset`` aligns a control dump that contains EARLIER updates
    (legacy probe-9 dumps; the new probe's capture_update makes offset
    unnecessary). Unequal lengths FAIL explicitly with both lengths —
    never a zip-truncate."""
    cf = control.get("frames", [])[offset:]
    cs = control.get("steps", [])[offset:]
    rf = resume.get("frames", [])
    rs = resume.get("steps", [])
    out: Dict[str, Any] = {
        "match": False,
        "control_frames": len(cf),
        "resume_frames": len(rf),
        "control_steps": len(cs),
        "resume_steps": len(rs),
        "first_diff": None,
        "fail": None,
    }
    if len(cf) != len(rf):
        out["fail"] = (
            f"control_frames={len(cf)} != resume_frames={len(rf)} — "
            "misaligned dumps must be aligned explicitly, never "
            "zip-truncated")
        return out
    if len(cs) != len(rs):
        out["fail"] = (
            f"control_steps={len(cs)} != resume_steps={len(rs)} — "
            "misaligned dumps must be aligned explicitly, never "
            "zip-truncated")
        return out
    for i in range(len(cf)):
        for field in FRAME_ORDER:
            if cf[i].get(field) != rf[i].get(field):
                out["first_diff"] = {
                    "kind": "frame", "index": i, "field": field}
                return out
    for i in range(len(cs)):
        c_slots = {s["slot"] for s in cs[i]} if isinstance(cs[i], list) else {cs[i]["slot"]}
        r_slots = {s["slot"] for s in rs[i]} if isinstance(rs[i], list) else {rs[i]["slot"]}
        if c_slots != r_slots:
            out["fail"] = (
                f"step {i} slot sets differ: control={sorted(c_slots)} "
                f"resume={sorted(r_slots)}")
            return out
        for slot in sorted(c_slots):
            c_step = (cs[i][slot] if isinstance(cs[i], list) else cs[i])
            r_step = (rs[i][slot] if isinstance(rs[i], list) else rs[i])
            for field in STEP_ORDER:
                if c_step.get(field) != r_step.get(field):
                    out["first_diff"] = {
                        "kind": "step", "index": i,
                        "field": f"slots.{slot}.{field}"}
                    return out
    out["match"] = True
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmp = sub.add_parser("compare", help="first-diff two probe artifacts")
    cmp.add_argument("--control", required=True, type=Path)
    cmp.add_argument("--resume", required=True, type=Path)
    cmp.add_argument("--offset", type=int, default=0,
                     help="control-side leading frames to skip (legacy "
                          "concatenated dumps)")
    args = parser.parse_args(argv)
    out = compare_dumps(
        json.loads(args.control.read_text()),
        json.loads(args.resume.read_text()),
        offset=args.offset,
    )
    print(json.dumps(out, indent=1))
    return 0 if out["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
