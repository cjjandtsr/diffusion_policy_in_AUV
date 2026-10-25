"""g6_gate_report.py — build the structured G6 gate evidence report.

Mirrors the G3 precedent (scripts/g3_10k_reset_gate_audit.py ->
program/reports/routeA_rl_env_v1/g3_gate/G3_GATE_REPORT.json): a single
JSON the design §5.8 / RTM audit trail can consume without parsing logs.

Inputs (all produced by the G6 evidence run):
  --test-log  the pytest -s output of test_g6_real_one_update_gpu.py
              (its three evidence prints: per-denoise k3 KL, clipfrac,
              resume parity summary)
  --probe-dir the g6 probe outdir (stage1/stage2 summaries + the three
              checkpoint counter values, written by
              g6_one_update_resume_probe.py)

Output: the report JSON at --out (default mirrors G3's location).

The nine §5.8 gates are reported from the test's own assertion results:
the test passing is the gate; this collector records the evidence values
the test printed, plus the watch-item bound and the memory finding
(minibatch 512 OOM on the 8 GiB card -> gate batch 128).

Run:
  python scripts/g6_gate_report.py --test-log temp/g6_test_run2.log \
      --probe-dir temp/g6_probe --out program/.../g6_gate/G6_GATE_REPORT.json
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

RATIO_SANITY_CEILING = 10.0  # pathological-recompute guard, not a bias bound
KL_FLOOR = 1e-4


def _find_line(text: str, prefix: str) -> Optional[str]:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return None


_SPECIAL_NUMBERS = {"nan": float("nan"), "inf": float("inf")}


def _eval_literal(node: Any) -> Any:
    """Strict literal evaluation of the gate's printed dict values."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SPECIAL_NUMBERS:
            return _SPECIAL_NUMBERS[node.id]
        raise ValueError(f"unexpected name: {node.id!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_literal(node.operand)
    if isinstance(node, ast.Dict):
        return {_eval_literal(k): _eval_literal(v)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [_eval_literal(e) for e in node.elts]
        return vals if isinstance(node, ast.List) else tuple(vals)
    raise ValueError(f"unsupported node: {ast.dump(node)}")


def _literal(text: str) -> Any:
    """literal_eval that also accepts bare nan/inf (the gate prints them)."""
    return _eval_literal(ast.parse(text, mode="eval").body)


def parse_test_log(log_text: str) -> Dict[str, Any]:
    """Extract the three structured evidence prints the gate test emits."""
    pdk_line = _find_line(log_text, "G6 per-denoise k3 KL:")
    clip_line = _find_line(log_text, "G6 clipfrac:")
    parity_line = _find_line(log_text, "G6 resume parity OK:")
    assert pdk_line, "no per-denoise KL print in test log (test failed?)"
    pdk = _literal(pdk_line.split("G6 per-denoise k3 KL: ", 1)[1])
    clip = _literal(clip_line.split("G6 clipfrac: ", 1)[1])
    m = re.search(r"(\d+) metric groups match within rtol=([\d.e-]+) "
                  r"atol=([\d.e-]+)", parity_line or "")
    assert m, "no resume-parity print in test log"
    active_kl = [v for v in pdk.values() if v == v]  # drop nan slots
    return {
        "per_denoise_k3_kl": pdk,
        "per_denoise_clipfrac": clip,
        "per_denoise_k3_kl_max": max(active_kl),
        "per_denoise_k3_kl_min_active": min(active_kl),
        "ratio_sanity_ceiling": RATIO_SANITY_CEILING,
        "recompute_sane": max(active_kl) < RATIO_SANITY_CEILING,
        "ratio_moved": max(active_kl) > KL_FLOOR,
        "resume_parity_groups": int(m.group(1)),
        "resume_parity_rtol": float(m.group(2)),
        "resume_parity_atol": float(m.group(3)),
    }


def parse_probe(probe_dir: Path) -> Dict[str, Any]:
    """Checkpoint counters from the probe's chained-resume evidence."""
    summary = json.loads((probe_dir / "stage2_summary.json").read_text())
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder

    ckpts = [summary["ckpt1"], summary["ckpt2"], summary["ckpt3"]]
    counts = [CheckpointBuilder.load(p).ppo_update_count for p in ckpts]
    env_steps = [
        CheckpointBuilder.load(p).environment_transition_count for p in ckpts
    ]
    return {
        "ppo_update_count_chain": counts,
        "env_step_accumulation": env_steps,
        "counter_accumulation_ok": counts == [1, 2, 3]
        and env_steps == sorted(env_steps)
        and len(set(env_steps)) == 3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-log", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    test = parse_test_log(Path(args.test_log).read_text())
    probe = parse_probe(Path(args.probe_dir))
    report: Dict[str, Any] = {
        "gate": "G6",
        "gate_source": "test_g6_real_one_update_gpu.py + probe",
        "minibatch": 128,
        "minibatch_official_default_512_infeasible_8GiB": True,
        "memory_finding": "official-default minibatch 512 OOMs the 8 GiB "
        "card (multi_stage_pointnet.py:39, 6 GiB cat during logprob "
        "recompute via shared PolicyObsEncoder); gate batch 128, §5.8 "
        "requires only single/deterministic minibatch order + K_epochs=1",
        "gates": {
            "g1_metrics_parameter_finite": "test-asserted",
            "g2_actor_grad_norm_gt_0": "test-asserted",
            "g3_critic_grad_norm_gt_0": "test-asserted",
            "g4_lora_update_norm_gt_0": "test-asserted",
            "g5_frozen_base_unet_update_norm_0": "test-asserted",
            "g6_unexecuted_ppo_grad_leq_1e-8": "test-asserted",
            "g7_ratio_ne_1": test["ratio_moved"],
            "g8_optimizer_state_survives_save_load": "test-asserted",
            "g9_same_seed_resume_parity": "test-asserted",
        },
        "watch_item": {
            "per_denoise_k3_kl_max": test["per_denoise_k3_kl_max"],
            "ratio_sanity_ceiling": test["ratio_sanity_ceiling"],
            "recompute_sane": test["recompute_sane"],
            "moved": test["ratio_moved"],
            "batch_bias_bounded_by": "G5 no-update identity measurement "
            "(4.9e-4 logp @B=128, ~0.08 amplified) — the per-denoise KL "
            "here is weight-movement signal + bias and is not a bias bound",
        },
        "resume_parity": {
            "groups_compared": test["resume_parity_groups"],
            "rtol": test["resume_parity_rtol"],
            "atol": test["resume_parity_atol"],
        },
        "probe": probe,
        "status": (
            "PASS"
            if (test["recompute_sane"] and test["ratio_moved"]
                and probe["counter_accumulation_ok"])
            else "FAIL"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
