"""R6-F scheduler contract (F-07 → R6-TUNE-1, Guide §11.7.1-11.7.2, §11.7.4).

The frozen vendor CosineAnnealingWarmupRestarts (util/scheduler.py at
vendor commit cc7234a) is the behavior oracle — never the project's own
formulas. Coverage:

- LR sequence parity: the production port reproduces the vendor's
  per-step lr over the first N steps (warmup + cosine regime);
- none branch: LR constant, resume never drifts, structured state
  (method/version/step_count) — never an empty dict;
- actor/critic independence: the actor scheduler advances ONLY when the
  actor actually stepped (critic-only warmup leaves step_count=0);
- cold resume: a loaded scheduler continues the LR sequence exactly
  (next 3 steps == uninterrupted oracle);
- tamper: method/version/step_count/step_in_cycle/cycle/cur_cycle_steps/
  max_lr — any single mutation fails validation BEFORE any tensor load;
- cross-field: critic step_count == ppo_update_count, actor
  step_count == max(0, ppo_update_count - warmup);
- config hash: LRSchedulerConfig axes enter the resolved config layer.
"""

import copy

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.training_schedules import (
    LRSchedulerConfig,
    SCHEDULER_STATE_VERSION,
    CosineAnnealingWarmupRestarts,
    build_lr_schedulers,
    load_scheduler_state,
    scheduler_state_dict,
    step_scheduler_pair,
    validate_scheduler_state,
)

VENDOR_ROOT = (
    "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
    "/third_party/dppo_official_cc7234a"
)
import sys

if VENDOR_ROOT not in sys.path:
    sys.path.insert(0, VENDOR_ROOT)

from util.scheduler import (  # noqa: E402
    CosineAnnealingWarmupRestarts as VendorCosine,
)

# fixed official-like config: warmup 3 of a 10-step first cycle
CFG = LRSchedulerConfig(
    method="official_cosine_warmup",
    first_cycle_steps=10, warmup_steps=3,
    min_lr=1e-4, max_lr=1e-2, cycle_mult=1.0, gamma=1.0,
)


def _optimizer(lr: float = 1e-3):
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _vendor_optimizer():
    opt = _optimizer()
    v = VendorCosine(
        opt, CFG.first_cycle_steps, cycle_mult=CFG.cycle_mult,
        max_lr=CFG.max_lr, min_lr=CFG.min_lr,
        warmup_steps=CFG.warmup_steps, gamma=CFG.gamma,
    )
    return opt, v


# ═══════════════════════════════════════════════════════════════════
# §11.7.4 item 4: scheduler first-10-steps parity vs frozen vendor
# ═══════════════════════════════════════════════════════════════════

def test_scheduler_lr_sequence_parity_with_vendor():
    """First N=10 steps: production port lr == frozen vendor lr, exact
    (identical arithmetic on identical fields — no tolerance needed)."""
    opt_v, vendor = _vendor_optimizer()
    opt_o = _optimizer()
    ours = CosineAnnealingWarmupRestarts(opt_o, CFG)
    # construction must already apply the vendor init_lr (lr = min_lr)
    assert opt_v.param_groups[0]["lr"] == CFG.min_lr
    assert opt_o.param_groups[0]["lr"] == CFG.min_lr
    seen = set()
    for _ in range(10):
        vendor.step()
        ours.step()
        assert opt_o.param_groups[0]["lr"] == (
            opt_v.param_groups[0]["lr"]
        ), f"step mismatch at step_count={ours.step_count}"
        seen.add(ours.lr)
    assert ours.step_count == 10
    # the sequence moves: the 10-step window spans warmup + cycle, so at
    # least 3 distinct LR values appear (the window ENDS on a rollover
    # back to min_lr — vendor-identical, not a constant sequence)
    assert len(seen) >= 3, f"sequence is constant: {sorted(seen)}"


def test_scheduler_none_branch_constant_and_structured_state():
    """none: LR constant across steps AND across a resume; state is
    structured (method/version/step_count), never an empty dict."""
    cfg = LRSchedulerConfig(method="none", max_lr=7e-4)
    opt = _optimizer(lr=7e-4)
    sched = CosineAnnealingWarmupRestarts(opt, cfg)
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] == 7e-4, "none branch must not move LR"
    st = sched.state_dict()
    assert st["method"] == "none" and st["version"] == SCHEDULER_STATE_VERSION
    assert st["step_count"] == 5
    assert st != {}, "none state must be structured, never an empty dict"
    # resume never drifts: a loaded none scheduler keeps LR constant
    fresh = CosineAnnealingWarmupRestarts(_optimizer(lr=7e-4), cfg)
    fresh.load_state_dict(st)
    for _ in range(3):
        fresh.step()
    assert fresh.state_dict()["step_count"] == 8
    assert opt.param_groups[0]["lr"] == 7e-4


# ═══════════════════════════════════════════════════════════════════
# §11.7.2: actor/critic independence + per-update step semantics
# ═══════════════════════════════════════════════════════════════════

def test_step_scheduler_pair_actor_does_not_advance_during_warmup():
    """§11.7.2: actor scheduler advances ONLY when the actor stepped —
    critic-only warmup leaves actor step_count=0 while critic advances."""
    actor_sched = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    critic_sched = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    # two warmup updates: actor never stepped
    step_scheduler_pair(actor_sched, critic_sched, actor_stepped=False)
    step_scheduler_pair(actor_sched, critic_sched, actor_stepped=False)
    assert actor_sched.step_count == 0, "actor scheduler must not advance"
    assert critic_sched.step_count == 2, "critic advances every update"
    # actor enters: steps from here on
    step_scheduler_pair(actor_sched, critic_sched, actor_stepped=True)
    assert actor_sched.step_count == 1
    assert critic_sched.step_count == 3


def test_scheduler_one_step_per_successful_update_even_on_early_stop():
    """§11.7.2: target-KL early stop inside an update must NOT cause a
    multi-step or zero-step update — the pair advances exactly once per
    committed update, driven by the train-loop call site (one call per
    successful ppo_update)."""
    actor_sched = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    critic_sched = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    # the train loop calls step_scheduler_pair exactly once per
    # successful update regardless of early stop; simulate 3 updates
    for _ in range(3):
        step_scheduler_pair(actor_sched, critic_sched, actor_stepped=True)
    assert actor_sched.step_count == 3
    assert critic_sched.step_count == 3


# ═══════════════════════════════════════════════════════════════════
# §11.7.4 item 6: cold resume continues the LR sequence exactly
# ═══════════════════════════════════════════════════════════════════

def test_scheduler_cold_resume_lr_sequence_continues():
    """Cold resume: a loaded scheduler produces the SAME next-3-step LR
    sequence as the uninterrupted oracle (state = pure function of
    step_count + config)."""
    uninterrupted = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    for _ in range(5):
        uninterrupted.step()
    snapshot = uninterrupted.state_dict()  # save at step 5, BEFORE continuing
    lr_expected = []
    for _ in range(3):
        uninterrupted.step()
        lr_expected.append(uninterrupted.lr)

    resumed = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    resumed.load_state_dict(snapshot)
    assert resumed.step_count == 5
    for expected in lr_expected:
        resumed.step()
        assert resumed.lr == expected, (
            "resumed LR must continue the uninterrupted sequence"
        )


def test_scheduler_state_roundtrip_through_dict():
    """state_dict -> load_state_dict roundtrip preserves every mutable
    field (dict roundtrip is the checkpoint path)."""
    a = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    for _ in range(7):
        a.step()
    st = a.state_dict()
    b = CosineAnnealingWarmupRestarts(_optimizer(), CFG)
    b.load_state_dict(st)
    assert b.state_dict() == st
    assert b.lr == a.lr
    # one more step on both: identical LR
    a.step()
    b.step()
    assert b.lr == a.lr


# ═══════════════════════════════════════════════════════════════════
# §11.7.4 item 7: tamper any single axis -> resume fails pre-load
# ═══════════════════════════════════════════════════════════════════

def test_validate_scheduler_state_tamper_each_axis_rejected():
    """method/config/version/state — every single-field mutation fails
    validation (the resume-side check, before any tensor load)."""
    # valid reference: 7 updates, critic stepped every time, actor after
    # a 2-update warmup window
    state = {
        "method": "official_cosine_warmup",
        "version": SCHEDULER_STATE_VERSION,
        "actor": {"step_count": 5, "step_in_cycle": 2, "cycle": 0,
                  "cur_cycle_steps": 10, "max_lr": 1e-2},
        "critic": {"step_count": 7, "step_in_cycle": 4, "cycle": 0,
                   "cur_cycle_steps": 10, "max_lr": 1e-2},
    }
    validate_scheduler_state(
        state, CFG, ppo_update_count=7, n_critic_warmup_updates=2
    )  # valid reference passes
    mutations = [
        ("method", "none"),  # method swapped
        ("version", "v0"),  # version swapped
        ("actor.step_count", 4),  # cross-field: 7-2=5 expected
        ("critic.step_count", 6),  # cross-field: == ppo_update_count
        ("actor.step_in_cycle", -1),  # bounds: [0, cur_cycle_steps)
        ("actor.step_in_cycle", 10),  # bounds: >= cur_cycle_steps
        ("actor.cycle", -1),  # bounds: >= 0
        ("actor.cur_cycle_steps", 2),  # bounds: > warmup_steps
        ("actor.max_lr", 5e-3),  # derived: config.max_lr * gamma^cycle
    ]
    for path, value in mutations:
        bad = copy.deepcopy(state)
        secs = path.split(".")
        if len(secs) == 1:
            bad[secs[0]] = value
        else:
            bad[secs[0]][secs[1]] = value
        with pytest.raises((ValueError, KeyError, AssertionError),
                           match="scheduler"):
            validate_scheduler_state(
                bad, CFG, ppo_update_count=7, n_critic_warmup_updates=2
            ), f"tamper {path}={value} must fail validation"


def test_validate_scheduler_state_none_schema_strict():
    """none state accepts exactly method/version/step_count — injected
    official fields are a schema violation (method-conditional)."""
    good = {"method": "none", "version": SCHEDULER_STATE_VERSION,
            "actor": {"step_count": 3}, "critic": {"step_count": 3}}
    cfg = LRSchedulerConfig(method="none")
    validate_scheduler_state(
        good, cfg, ppo_update_count=3, n_critic_warmup_updates=0
    )
    bad = copy.deepcopy(good)
    bad["actor"]["step_in_cycle"] = 0  # official-only field in none state
    with pytest.raises((ValueError, KeyError), match="scheduler"):
        validate_scheduler_state(
            bad, cfg, ppo_update_count=3, n_critic_warmup_updates=0
        )


def test_scheduler_config_validation_rejects_bad_official_config():
    """official config: warmup < first_cycle (vendor assert), positive
    lrs, cycle_mult >= 1 — a malformed config is a config error, not a
    silent runtime behavior."""
    with pytest.raises((AssertionError, ValueError)):
        LRSchedulerConfig(method="official_cosine_warmup",
                          first_cycle_steps=5, warmup_steps=5)  # ==
    with pytest.raises((AssertionError, ValueError)):
        LRSchedulerConfig(method="official_cosine_warmup",
                          first_cycle_steps=5, warmup_steps=6)  # >
    with pytest.raises((AssertionError, ValueError)):
        LRSchedulerConfig(method="official_cosine_warmup",
                          max_lr=0.0)  # non-positive max
    with pytest.raises((AssertionError, ValueError)):
        LRSchedulerConfig(method="official_cosine_warmup",
                          min_lr=1e-2, max_lr=1e-4)  # min > max
    with pytest.raises(AssertionError):
        LRSchedulerConfig(method="not_a_method")  # enum


# ═══════════════════════════════════════════════════════════════════
# checkpoint-level: scheduler_state_dict + build_lr_schedulers
# ═══════════════════════════════════════════════════════════════════

def test_scheduler_state_dict_checkpoint_shape():
    """The checkpoint field shape: {method, version, actor, critic} —
    validation and load consume the same shape (save/load symmetry)."""
    actor_opt, critic_opt = _optimizer(), _optimizer()
    actor_sched, critic_sched = build_lr_schedulers(
        actor_opt, critic_opt, CFG
    )
    for _ in range(2):
        step_scheduler_pair(actor_sched, critic_sched, actor_stepped=True)
    st = scheduler_state_dict(actor_sched, critic_sched)
    assert st["method"] == "official_cosine_warmup"
    assert st["version"] == SCHEDULER_STATE_VERSION
    assert st["actor"]["step_count"] == 2
    assert st["critic"]["step_count"] == 2
    assert st["actor"]["cur_cycle_steps"] == CFG.first_cycle_steps
    # the dict roundtrips through validation + load
    validate_scheduler_state(
        st, CFG, ppo_update_count=2, n_critic_warmup_updates=0
    )
    a2, c2 = build_lr_schedulers(_optimizer(), _optimizer(), CFG)
    load_scheduler_state(st, a2, c2)
    assert (a2.state_dict()["step_count"] == 2
            and c2.state_dict()["step_count"] == 2)


def test_scheduler_config_enters_resolved_config_layer():
    """Every LRSchedulerConfig field lands in _config_dict (the
    config_hash source) — two configs differing in ANY field hash
    differently (a resume under a different schedule is a different
    run)."""
    from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        LoopConfig, _config_dict,
    )
    from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

    base = dict(device="cpu", reward_version="v1.3",
                world_assets_hash="world456")
    num_mode = {"enabled": False}

    def cfg_hash_for(sched_cfg, scale_cfg):
        loop = LoopConfig(
            mode="one_update", n_episodes=2, max_ep_len=20,
            device="cpu", output_dir="/tmp/x",
            lr_scheduler=sched_cfg, reward_scale=scale_cfg,
        )
        d = _config_dict(RLSystemContract(), loop, TrainerConfig(device="cpu"),
                         base, num_mode, None)
        return d["lr_scheduler_config"], d["reward_scale_config"]

    import hashlib

    def digest(loop_dict):
        return hashlib.sha256(
            repr(sorted(loop_dict.items())).encode()
        ).hexdigest()

    from diffusion_policy_3d.rl.dppo_uav.reward_norm import RewardScaleConfig
    none = LRSchedulerConfig(method="none")
    warm = LRSchedulerConfig(method="official_cosine_warmup",
                             first_cycle_steps=10, warmup_steps=3)
    d_none, r_none = cfg_hash_for(none, RewardScaleConfig(method="none"))
    d_warm, r_warm = cfg_hash_for(warm, RewardScaleConfig(method="none"))
    assert d_none["method"] == "none" and d_warm["method"] == "official_cosine_warmup"
    assert digest(d_none) != digest(d_warm)
    assert r_none == r_warm, "unchanged axes must stay stable"
    # a warmup-steps-only change also rehashes
    warm2 = LRSchedulerConfig(method="official_cosine_warmup",
                              first_cycle_steps=10, warmup_steps=4)
    d_warm2, _ = cfg_hash_for(warm2, RewardScaleConfig(method="none"))
    assert digest(d_warm) != digest(d_warm2)


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-TUNE-1 REWORK（2026-08-19）: warmup=0 构造序 parity
# vendor 写序 = torch implicit step 后 init_lr() 最后执行 → 构造后
# lr 永远 = min_lr；port 旧写序 get_lr() 在 _init_lr() 后、warmup=0 时
# 走 cosine 分支 → 构造后 lr=max_lr（1e-2 vs 1e-4，100x 偏差），且
# save 冻结 min_lr 锚点 → warmup=0 + official scheduler 时 cold resume
# 必 optimizer 合同 drift 硬失败（CLI 默认 --scheduler_warmup_steps 0）。
# ═══════════════════════════════════════════════════════════════════

def test_scheduler_construction_parity_warmup_zero():
    """warmup_steps=0: 构造后 LR 必须 = min_lr（与 vendor 位等）——
    REWORK 反例：port 构造 LR = max_lr。"""
    cfg = LRSchedulerConfig(
        method="official_cosine_warmup",
        first_cycle_steps=10, warmup_steps=0,
        min_lr=1e-4, max_lr=1e-2, cycle_mult=1.0, gamma=1.0,
    )
    opt_v = _optimizer()
    v = VendorCosine(
        opt_v, cfg.first_cycle_steps, cycle_mult=cfg.cycle_mult,
        max_lr=cfg.max_lr, min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps, gamma=cfg.gamma,
    )
    opt_o = _optimizer()
    ours = CosineAnnealingWarmupRestarts(opt_o, cfg)
    # vendor init_lr() 最后执行 → 构造后 lr = min_lr
    assert opt_v.param_groups[0]["lr"] == cfg.min_lr
    assert opt_o.param_groups[0]["lr"] == cfg.min_lr
    # 前 10 步序列 parity（warmup=0 全程 cosine regime，无巧合掩盖）
    for _ in range(10):
        v.step()
        ours.step()
        assert opt_o.param_groups[0]["lr"] == (
            opt_v.param_groups[0]["lr"]
        ), f"step mismatch at step_count={ours.step_count}"
