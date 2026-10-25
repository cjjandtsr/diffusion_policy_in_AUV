"""G6 (design §5.8): one-update mechanism test on the REAL env + BC policy.

One PPO batch update over a fixed-seed real rollout (32 episodes, attempt-
index seeding, no reward ablation), then every §5.8 gate is re-verified
independently of the loop's own invariants:

  - loss / gradient / parameter all finite;
  - actor grad norm > 0;  critic grad norm > 0;  LoRA update norm > 0;
  - frozen base UNet update norm == 0 (N1 boundary: only LoRA moves);
  - unexecuted-action PPO gradient norm <= 1e-8 (F10 ghost-action mask);
  - at least one denoising step ratio != 1 (post-update recompute);
  - optimizer state survives save/load (state_dict parity after restore);
  - same-seed resume reproduces the next-update metrics within tolerance
    (N2 boundary: the checkpoint holds the live trained actor + optimizer
    + RNG states, so a fresh world restored from it runs update 2
    identically to the continuous world that never stopped).

watch item (R7-3 actual-batch semantics, guide Task R7-3 Step 3/6):
ppo_update runs ONE official facade.loss (vendor PPODiffusion.loss) call
per minibatch at the ACTUAL minibatch batch — B=128 production — and
the G5 no-update identity (test_actual_batch_logprob_identity) gates
the B=128 recompute vs stored B=1 old logprobs at <= 1e-5. The
batch-bias question of the old microbatch era is therefore resolved by
the actual-batch gate, not by recomputing at B=1. The per-denoise KL
after the update is pure intra-update weight movement — the PPO signal
§5.8 gate 7 asks for ("at least one denoising step ratio != 1 after
the update"). The gate does NOT assert the G5 1e-5 identity after an
update (weights moved); the measured KL is recorded as evidence.
RATIO_SANITY_CEILING only guards pathological recompute breakage
(unreachable by a healthy lr=1e-3 one-update signal).

R7-3 Step 6 (guide): this test also proves the vendor loss ACTUALLY
ran at actual B=128 — a class-level spy on OfficialPhase1aFacade.loss
(counts calls; every call's minibatch size read from the facade's
observer cache, the same cache the trainer's observer consumes) — and
that the actor optimizer owns exactly the LoRA tensors (AdamW state
entry count == LoRA tensor count; base UNet never in the optimizer).

[G6 finding 1] minibatch B=128, not the official-default 512: the first
gate run OOM'd at 512 (multi_stage_pointnet.py:39, 6 GiB single cat on
the 8 GiB RTX 4060 — the logprob recompute's shared PolicyObsEncoder
pointnet at 512). §5.8 only requires "single minibatch or deterministic
minibatch order" + K_epochs=1, so 128 (proven memory-safe in G5's own
B=128 identity measurement) is the gate's batch; the official-default
512 is recorded as infeasible on this card for T6 config selection.

[G6 finding 2] run 2 FAIL was a gate-semantics bug in this test, not a
code defect: RATIO_BIAS_UBOUND=0.15 sat on the combined signal+bias
(measured 0.2344 at denoising step 2 on update 1) and rejected the
healthy weight-movement signal. Corrected to §5.8's literal requirement
(ratio != 1) + a sanity ceiling; the correction is recorded in
task_plan_t5_rework.md, not silently relaxed.

A PASS here IS the G6 evidence; a FAIL is a production gap repaired
test-first (resume support was absent at the time of writing).

Guide R4.3 Step 2 lists "no-update identity 先满足 G5 阈值" as a G6
gate input: that clause is satisfied by GATE ORDER — G5 v2 (R4.2) ran
the same production no-update path and asserted the identity before G6
was re-gated (review_R4_part1 P2-2). The G6 test does not re-verify
no-update identity on its own buffer (weights moved by the update; the
pre-update identity is G5's gate).

R1.4 extension (episode-boundary GAE): the real fixture must contain at
least one terminated episode, one truncated episode, and one plain batch
cut (the corrected compute_gae's three transition types, R1.3); resume
parity additionally compares the corrected advantages/returns hashes
(metrics.advantages_hash/returns_hash, trainer.py) bitwise across the
continuous vs resumed worlds.

Run: pytest -m gpu tests/dppo_parity/test_g6_real_one_update_gpu.py
Skipped (not failed) when CUDA or the production assets are unavailable.
"""

import math
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from diffusion_policy_3d.policy.lora_adapters import BUNDLE_ATTR
from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
)
from diffusion_policy_3d.rl.dppo_uav.numerics import configure_numerical_mode
from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
    OfficialPhase1aFacade,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    FORMAL_NUMERICAL_MODE,
    LoopConfig,
    restore_training_state,
    resume_identity_for,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

G6_EPISODES = 32
MAX_EP_LEN = 60
SEED = 0  # fixed construction + rollout seed for both fresh worlds
# Gate-9 compares World 2's continuous next update vs World 1's resumed
# update under the SAME episode sequence — every call site builds a FRESH
# sampler with the same base_seed so both sides replay the identical
# sampler sequence (checkpoint-persisted sampler state is R3).
G6_SAMPLER_BASE_SEED = 1

# watch item (R7-3 actual-batch semantics, guide Task R7-3 Step 3/6):
# ppo_update recomputes the likelihood at the ACTUAL minibatch — B=128,
# one official facade.loss call per minibatch (the microbatch=1 loop is
# deleted) — so the post-update per-denoise KL is the pure weight-
# movement signal of §5.8 gate 7 ("at least one denoising step ratio
# != 1 after the update"). The B=1-vs-B=128 batch-bias question is
# resolved by G5's actual-batch no-update identity gate, NOT by this
# KL. KL_FLOOR enforces the "ratio != 1" requirement;
# RATIO_SANITY_CEILING only catches a pathological recompute breakage
# (healthy lr=1e-3 one-update signal stays << 10) — it is not a bias
# bound.
RATIO_SANITY_CEILING = 10.0
KL_FLOOR = 1e-4  # at least one denoising step must move

# resume parity tolerance (same op sequence on the same GPU is bit-stable
# for the relevant magnitudes; keep a small slack for any kernel scheduling
# differences between the two worlds)
RESUME_RTOL = 1e-4
RESUME_ATOL = 1e-6

# R7-1 (2026-08-22 user ruling): base actor = E5 retrain e5_lambda0p5
# (V11-fixed collision loss, lambda_col=0.5, sha256 c1fca4ac…) — same
# default as conftest.DEFAULT_BC_CKPT; the loader allowlist is bound to
# this checkpoint's hash.
DEFAULT_BC_CKPT = (
    "data/outputs/2026.08.16/08.53.26_e5_lambda0p5"
    "/checkpoints/latest.ckpt"
)


# R6-C (§11.4.1): the executed/unexecuted ratio is an OptionalStatistic
# following the two-state contract (metrics.py
# executed_unexecuted_ratio_statistic) — with the F10 ghost-action mask
# the unexecuted gradient is IDENTICALLY 0, so the trainer records the
# null state (sample_count=0/value=None/valid=False/
# zero_executed_gradient_denominator) instead of the old NaN sentinel
# (R5.8-F H-2) or a ~1e8 pseudo-signal. The null semantics are asserted
# by the F10 gate below; the finite check treats None as the legitimate
# null value (never a defect).


def _finite_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Names of numeric metric fields that are not finite.

    OptionalStatistic values are None (legitimate null state) or a
    finite float (the factory raises on non-finite) — never NaN/Inf.
    """
    bad: List[str] = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)) and not np.isfinite(v2):
                    bad.append(f"{k}.{k2}={v2}")
        elif isinstance(v, (int, float)) and not np.isfinite(v):
            bad.append(f"{k}={v}")
    return bad


def _compare_value(parent_value: Any, child_value: Any, path: str,
                   mismatches: List[str]) -> None:
    """Recursive two-state structural comparison (R6-C §11.4.1).

    Handles every value shape the trainer metrics dict can carry:
    scalars (isclose), None (presence-compare), dicts (recurse over
    the key union — a key present on one side only is a structural
    divergence), and lists (element-wise). Never coerces a non-number
    through float(): dict leaves (OptionalStatistic records, per-role
    maps), str leaves (invalid_reason, hashes, variance space) and
    list leaves (action_variance_by_dim) are legitimate metric shapes.
    (GPU RED 2026-08-18: the flat two-level loop crashed with
    float(None) on one-sided nulls, then float(dict) on the nested
    executed_vs_unexecuted_grad_norm.ratio / outcome_counts role maps
    once the None case was handled; str and list leaves were the next
    latent class — one recursive walker covers them all.) Both None ==
    equal; one-sided None == genuine divergence -> mismatch. The
    legacy NaN-sentinel skip is retained: non-finite numerics are an
    engine violation the strict writer catches at persist time.
    """
    if parent_value is None or child_value is None:
        if parent_value != child_value:
            mismatches.append(
                f"{path}: None vs {child_value!r}"
                if parent_value is None else
                f"{path}: {parent_value!r} vs None")
        return
    if isinstance(parent_value, dict) or isinstance(child_value, dict):
        if not (isinstance(parent_value, dict)
                and isinstance(child_value, dict)):
            mismatches.append(
                f"{path}: dict vs {type(child_value).__name__}")
            return
        for key in parent_value.keys() | child_value.keys():
            if key not in parent_value:
                mismatches.append(
                    f"{path}.{key}: <missing> vs {child_value[key]!r}")
                continue
            if key not in child_value:
                mismatches.append(
                    f"{path}.{key}: {parent_value[key]!r} vs <missing>")
                continue
            _compare_value(parent_value[key], child_value[key],
                           f"{path}.{key}", mismatches)
        return
    if isinstance(parent_value, (list, tuple)) or isinstance(
            child_value, (list, tuple)):
        if not (isinstance(parent_value, (list, tuple))
                and isinstance(child_value, (list, tuple))):
            mismatches.append(
                f"{path}: list vs {type(child_value).__name__}")
            return
        if len(parent_value) != len(child_value):
            mismatches.append(
                f"{path}: length {len(parent_value)} vs "
                f"{len(child_value)}")
            return
        for i, (pv, cv) in enumerate(zip(parent_value, child_value)):
            _compare_value(pv, cv, f"{path}[{i}]", mismatches)
        return
    # R6-C rework (P1-4): non-finite on EITHER side is a hard-fail —
    # NaN/Inf is not a legitimate "no data" representation under the
    # two-state contract (no data = OptionalStatistic count==0 with
    # value None). The old parent-side skip let engine violations
    # through as equality.
    if isinstance(parent_value, (int, float)) and not np.isfinite(
            parent_value):
        mismatches.append(f"{path}: parent non-finite {parent_value!r}")
        return
    if isinstance(child_value, (int, float)) and not np.isfinite(
            child_value):
        mismatches.append(f"{path}: child non-finite {child_value!r}")
        return
    if isinstance(parent_value, (int, float)) and isinstance(
            child_value, (int, float)):
        if not math.isclose(
            float(child_value), float(parent_value),
            rel_tol=RESUME_RTOL, abs_tol=RESUME_ATOL,
        ):
            mismatches.append(f"{path}: {parent_value} vs {child_value}")
        return
    # Non-numeric scalars (str tags, hashes, ...): strict equality — a
    # differing tag is a genuine parity break.
    if parent_value != child_value:
        mismatches.append(f"{path}: {parent_value!r} vs {child_value!r}")


def _compare_metrics(parent: Dict[str, Any], child: Dict[str, Any],
                     path: str) -> List[str]:
    """Resume-parity comparison of the trainer metrics dicts.

    R6-C (§11.4.1): the metrics dict carries mixed value shapes —
    scalars, OptionalStatistic records (dict), per-role/per-index maps
    (dict), distribution summaries (dict), a list (action_variance_
    by_dim) and str tags — compared structurally by _compare_value:
    parity breaks are reported as mismatches, never float() coercion
    of a non-number (GPU RED 2026-08-18: float(None) round 2,
    float(dict) round 3).
    """
    mismatches: List[str] = []
    for k, v in parent.items():
        if k not in child:
            mismatches.append(f"{path}.{k}: <missing> in child")
            continue
        _compare_value(v, child[k], f"{path}.{k}", mismatches)
    for k in child:
        if k not in parent:
            mismatches.append(f"{path}.{k}: <missing> in parent")
    return mismatches


def _param_update_norm(params: Any) -> float:
    total = 0.0
    for p in params:
        total += float(p.detach().pow(2).sum().item())
    return math.sqrt(total)


def _fresh_world(device: str) -> Tuple[Any, Dict[str, Any], ValueCritic,
                                       torch.optim.Optimizer,
                                       torch.optim.Optimizer]:
    """A fresh BC policy + zero-init LoRA + E040 critic + optimizers.

    Deterministic construction: same seed sequence for both worlds, so the
    (tiny) xavier-init variance of the critic head cannot break resume
    parity — restore_training_state overwrites it anyway, but construction
    order must not consume RNG differently across worlds.
    """
    from diffusion_policy_3d.policy.lora_adapters import insert_lora
    from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
    from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy

    ckpt = os.environ.get("DPPO_BC_CKPT", DEFAULT_BC_CKPT)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    policy = load_ddpm_policy(ckpt, device=device)
    lora_info = insert_lora(policy.model, r=8, alpha=8, verbose=False)
    contract = RLSystemContract()
    torch.manual_seed(SEED)  # critic xavier init deterministic across worlds
    torch.cuda.manual_seed_all(SEED)
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
    # R7-3 Step 2: actor optimizer owns exactly the registered LoRA
    # bundle trainable parameters; AdamW (pre-registered R7 change —
    # the production entry uses the same construction).
    lora_bundle = policy.model.dppo_lora_adapters
    actor_params = tuple(lora_bundle.trainable_parameters())
    assert actor_params, "registered LoRA bundle has no trainable parameters"
    actor_optimizer = torch.optim.AdamW(actor_params, lr=1e-3)
    critic_optimizer = torch.optim.AdamW(
        critic.value_head_parameters(), lr=1e-3
    )
    return policy, lora_info, critic, actor_optimizer, critic_optimizer


@pytest.mark.gpu
def test_g6_one_update_mechanism(
    real_policy_lora, real_adapter, tmp_path, monkeypatch
):
    """§5.8 gates 1-9 over one real PPO update + resume parity."""
    _, _, device = real_policy_lora
    adapter, specs, contract = real_adapter

    runtime = {
        "device": device,
        "reward_version": "v16",
        "manifest_hash": "g6-gate",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "world_assets_hash": "?",
        "timestamp": "g6-gate",
    }
    trainer_config = TrainerConfig(batch_size=128, device=device)

    # ─────────────────────────────────────────────────────────────────
    # World 2 (continuous baseline): run1 -> run2 in one process
    # ─────────────────────────────────────────────────────────────────
    policy2, lora_info2, critic2, opt2a, opt2c = _fresh_world(device)

    # R7-3 Step 6: prove the vendor loss ACTUALLY ran at actual B=128 —
    # class-level spy on OfficialPhase1aFacade.loss. Every call's
    # minibatch size is read from the facade's observer cache (single
    # slot, replaced by the last get_logprobs_subsample of the loss call;
    # one loss call is one minibatch, so the batch is uniform within it).
    loss_calls = {"n": 0, "batches": []}
    # capture the ORIGINAL class method BEFORE patching — the spy must
    # call the pre-patch vendor loss, not itself (monkeypatch.setattr
    # rebinds the class attribute, so an in-spy class lookup would recurse)
    original_loss = OfficialPhase1aFacade.loss

    def spy_loss(self, *args, use_bc_loss=False, **kwargs):
        out = original_loss(
            self, *args, use_bc_loss=use_bc_loss, **kwargs
        )
        loss_calls["n"] += 1
        loss_calls["batches"].append(
            int(self.observer_last["denoising_inds"].shape[0])
        )
        return out

    monkeypatch.setattr(OfficialPhase1aFacade, "loss", spy_loss)

    lc1 = LoopConfig(
        mode="one_update", n_episodes=G6_EPISODES, max_ep_len=MAX_EP_LEN,
        device=device, output_dir=str(tmp_path / "world2_run1"),
    )
    # the registered LoRA bundle (BUNDLE_ATTR, lora_adapters.py:257) IS
    # visible in state_dict — exclude its keys so the diff measures ONLY
    # base-UNet movement (base is autograd-frozen by insert_lora :272-278)
    base_before = {
        k: v.detach().clone()
        for k, v in policy2.model.state_dict().items()
        if not k.startswith(BUNDLE_ATTR + ".")
    }
    res1 = run_e2e_batch(
        adapter, policy2, critic2, contract, lc1, trainer_config, specs,
        opt2a, opt2c, lora_info2, runtime,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=G6_SAMPLER_BASE_SEED
        ),
    )
    m1 = res1["metrics"]
    ckpt1 = res1["checkpoint_path"]
    assert ckpt1 is not None, "one_update must write a checkpoint"

    # ── The real fixture must exercise all three transition types
    # (terminated episode, truncated episode, plain batch cut) so the
    # corrected episode-boundary GAE is covered by the real env.
    buf1 = res1["buffer"]
    n_term = sum(1 for t in buf1.terminated if t)
    n_trunc = sum(1 for t in buf1.truncated if t)
    n_plain = sum(
        1 for t, r in zip(buf1.terminated, buf1.truncated) if not t and not r
    )
    print(f"G6 buffer coverage: terminated={n_term} truncated={n_trunc} "
          f"plain={n_plain} total={len(buf1)}")
    assert n_term >= 1, "G6 fixture lacks a terminated episode"
    assert n_trunc >= 1, "G6 fixture lacks a truncated episode"
    assert n_plain >= 1, "G6 fixture lacks a plain batch cut"

    # ── §5.8 gate 1: loss / gradient / parameter all finite ─────────
    bad = _finite_metrics(m1)
    assert not bad, f"non-finite metrics: {bad}"
    for p in list(opt2a.param_groups[0]["params"]) + list(
        critic2.value_head_parameters()
    ):
        assert bool(torch.isfinite(p).all()), "non-finite parameter"

    # ── §5.8 gates 2-4: actor / critic grads, LoRA update norm > 0 ──
    assert m1["actor_grad_norm"] > 0.0, m1["actor_grad_norm"]
    assert m1["critic_grad_norm"] > 0.0, m1["critic_grad_norm"]
    assert m1["parameter_update_norm"] > 0.0, m1["parameter_update_norm"]
    lora_update_norm = _param_update_norm(lora_info2["lora_params"])
    assert lora_update_norm > 0.0, lora_update_norm

    # ── §5.8 gate 5: frozen base UNet update norm == 0 ─────────
    # base_before above already excludes the LoRA bundle keys; base is
    # autograd-frozen, so this diff must read exactly 0.0.
    base_after = policy2.model.state_dict()
    base_update_norm = math.sqrt(
        sum(
            float((base_after[k].detach() - base_before[k]).pow(2).sum().item())
            for k in base_before
        )
    )
    assert base_update_norm == 0.0, f"base UNet moved: {base_update_norm}"

    # ── R7-3 Step 6 gates: vendor loss at actual B=128; actor optimizer
    # owns exactly the LoRA tensors (state entry count == LoRA tensor
    # count — base UNet never in the optimizer) ───────────────────────
    assert loss_calls["n"] >= 1, "vendor facade.loss never called"
    assert all(b == trainer_config.batch_size for b in loss_calls["batches"]), (
        f"vendor loss ran off the actual batch: {loss_calls['batches']}"
    )
    assert len(opt2a.state) == len(lora_info2["lora_params"]), (
        f"actor optimizer state entries {len(opt2a.state)} != "
        f"LoRA tensor count {len(lora_info2['lora_params'])}"
    )
    print(f"G6 vendor loss calls: {loss_calls}")

    # ── §5.8 gate 6: unexecuted-action PPO gradient <= 1e-8 (F10) ───
    unex = m1["executed_vs_unexecuted_grad_norm"]["unexecuted"]
    assert unex <= 1e-8, f"unexecuted grad norm {unex} > 1e-8 (mask regression)"
    # R6-C (§11.4.1) two-state semantics: the ratio is null iff the mask
    # zeroed the unexecuted grads — nothing else; the null state carries
    # the frozen zero_executed_gradient_denominator reason.
    ratio = m1["executed_vs_unexecuted_grad_norm"]["ratio"]
    null = ratio["value"] is None
    assert null == (unex <= 1e-8), (
        f"ratio two-state semantics broken: ratio={ratio} unexecuted={unex}"
    )
    if null:
        assert ratio["sample_count"] == 0 and ratio["valid"] is False
        assert (ratio["invalid_reason"]
                == "zero_executed_gradient_denominator"), ratio
    else:
        assert ratio["sample_count"] > 0 and ratio["valid"] is True
        assert math.isfinite(ratio["value"]), ratio

    # ── §5.8 gate 7 + watch item: ratio != 1 (weight-movement signal) ──
    # The per-denoise KL is the post-update recompute signal (see module
    # docstring); a sane ceiling guards against a broken recompute, it is
    # not a bias bound (G5's no-update identity measurement is).
    pdk = m1["per_denoise_approx_kl"]
    # R6-C (§11.4.1): per-denoise means are always finite (an empty or
    # non-finite index hard-fails in the trainer) — the finiteness
    # filter below is a defensive no-op, kept for the older contract.
    active = {k: v for k, v in pdk.items() if v == v}
    assert active, "no per-denoise KL recorded"
    assert max(active.values()) > KL_FLOOR, (
        f"no denoising step ratio != 1 (max k3 KL {max(active.values())})"
    )
    assert max(active.values()) < RATIO_SANITY_CEILING, (
        f"recompute pathological: max per-denoise KL "
        f"{max(active.values())} >= {RATIO_SANITY_CEILING}"
    )
    print(f"G6 per-denoise k3 KL: {pdk}")
    print(f"G6 clipfrac: {m1['per_denoise_clipfrac']}")

    # ── §5.8 gate 8: optimizer state survives save/load ─────────────
    saved = CheckpointBuilder.load(ckpt1)
    assert saved.actor_optimizer_state["state"], "ckpt holds no optimizer state"
    assert saved.value_optimizer_state["state"]

    # ─────────────────────────────────────────────────────────────────
    # World 1 (resume): fresh world, restore the run1 checkpoint.
    # ─────────────────────────────────────────────────────────────────
    policy1, lora_info1, critic1, opt1a, opt1c = _fresh_world(device)
    restored = restore_training_state(
        ckpt1, policy1, critic1, lora_info1, opt1a, opt1c,
        # R3.2 RESUME-HASH-GATE: the runtime identity must match the
        # checkpoint's, or the restore refuses before any state_dict load.
        runtime_identity=resume_identity_for(
            contract, lc1, trainer_config, runtime,
            configure_numerical_mode(FORMAL_NUMERICAL_MODE), policy1,
            # The training_identity axes must match the SAVE side (world
            # 2's run_e2e_batch): base_seed/n_envs/lora/optimizers —
            # otherwise the config_hash gate refuses the resume (P0-2).
            base_seed=G6_SAMPLER_BASE_SEED,
            n_envs=int(adapter.n_envs),
            lora_info=lora_info1,
            actor_optimizer=opt1a,
            critic_optimizer=opt1c,
        ),
    )
    assert restored["ppo_update_count"] == 1, restored
    assert restored["environment_transition_count"] == res1["env_steps"]

    # policy_state is saved as CPU clones and
    # DictOfTensorMixin._load_from_state_dict (vendor, frozen) REPLACES
    # params with them — the normalizer's scale/offset must NOT end up on
    # CPU, or the next PolicyObsEncoder forward (normalize casts to
    # scale.device) feeds CPU obs into the CUDA pointnet.
    anchor = next(policy1.model.parameters())
    for p in policy1.normalizer.parameters():
        assert p.device == anchor.device, (
            f"restore moved normalizer param to {p.device} "
            f"(policy anchor {anchor.device})"
        )

    # optimizer parity: the restored optimizer matches the continuous
    # world's — both measured at the same (post-update-1) instant, before
    # either world advances through update 2.
    sd1 = opt1a.state_dict()
    sd2 = opt2a.state_dict()
    for k in sd2["state"]:
        for mk, v in sd2["state"][k].items():
            if torch.is_tensor(v):
                w = sd1["state"][k][mk]
                assert torch.equal(v, w), (
                    f"optimizer state mismatch after restore: {k}.{mk}"
                )

    # ── continuous run2 (World 2 keeps going) ───────────────────────
    # resume_from=ckpt1 is an idempotent no-op here — the restored state
    # IS world2's live state (same process, nothing consumed in between)
    # — its only effect is carrying the counters so the continuing world
    # numbers its own next checkpoint "update 2".
    lc2 = LoopConfig(
        mode="one_update", n_episodes=G6_EPISODES, max_ep_len=MAX_EP_LEN,
        device=device, output_dir=str(tmp_path / "world2_run2"),
    )
    res2 = run_e2e_batch(
        adapter, policy2, critic2, contract, lc2, trainer_config, specs,
        opt2a, opt2c, lora_info2, runtime, resume_from=ckpt1,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=G6_SAMPLER_BASE_SEED
        ),
    )
    m2 = res2["metrics"]
    assert res2["checkpoint_path"] is not None
    assert CheckpointBuilder.load(res2["checkpoint_path"]).ppo_update_count == 2

    # ── resumed run2 (World 1, interrupted, restored from ckpt1) ────
    # Must reproduce World 2's continuous update-2 metrics (gate 9).
    lc_r = LoopConfig(
        mode="one_update", n_episodes=G6_EPISODES, max_ep_len=MAX_EP_LEN,
        device=device, output_dir=str(tmp_path / "world1_resume"),
    )
    res_r = run_e2e_batch(
        adapter, policy1, critic1, contract, lc_r, trainer_config, specs,
        opt1a, opt1c, lora_info1, runtime, resume_from=ckpt1,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=G6_SAMPLER_BASE_SEED
        ),
    )
    mr = res_r["metrics"]

    # ── §5.8 gate 9: same-seed resume reproduces next-update metrics ─
    # Two-state contract aware (§11.4.1): None (count==0) on either side
    # is compared for presence, never coerced through float().
    mismatches = _compare_metrics(m2, mr, "metrics")
    assert not mismatches, (
        "resume update-2 metrics deviate from the continuous world: "
        + "; ".join(mismatches[:8])
    )
    # ── Corrected-GAE resume parity — the advantages/returns the update
    # consumed must be bit-identical across worlds (per-transition
    # bootstrap + episode boundaries). The numeric metric loop above skips
    # these string fields; compare them explicitly.
    assert m2["advantages_hash"] == mr["advantages_hash"], (
        "resume advantages hash mismatch: "
        f"{m2['advantages_hash']} vs {mr['advantages_hash']}"
    )
    assert m2["returns_hash"] == mr["returns_hash"], (
        "resume returns hash mismatch: "
        f"{m2['returns_hash']} vs {mr['returns_hash']}"
    )
    print(f"G6 resume parity OK: {len(m2)} metric groups + GAE hashes match "
          f"within rtol={RESUME_RTOL} atol={RESUME_ATOL}")
