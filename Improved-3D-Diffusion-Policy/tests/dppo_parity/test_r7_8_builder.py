"""R7-8 Step 2/3/5: quota selection, train-leakage check and aggregate
computation of the bank builder.

Quota (Step 2): exactly 100 episodes, each main capability >= 20,
each main role >= 20, unknown/no-opportunity-only <= 10. Leakage
(Step 3): bank ∩ train manifest = 0, hard-fail. Selection (Step 5):
deterministic manifest-order, never conditioned on any RL checkpoint
result — the candidates list carries no RL signal by construction.
"""

import pytest

from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
    BANK_SIZE,
    CAPABILITY_ROLES,
    MIN_EPISODES_PER_ROLE,
    check_train_leakage,
    compute_aggregate,
    expand_eval_candidates,
    freeze_and_annotate,
    select_episodes,
)


# ═══════════════════════════════════════════════════════════════════
# Step 2 quota selection
# ═══════════════════════════════════════════════════════════════════

def _candidates(n_anti: int, n_support: int, n_dense: int,
                extra_roles: tuple = ()) -> list:
    """(world_uid, role, seed) triples in manifest order, first come
    first served per role."""
    out = []
    for role, n in (("anticipatory", n_anti),
                    ("support_expansion", n_support),
                    ("dense_recovery", n_dense)):
        out.extend((f"w_{role}_{i}", role, i) for i in range(n))
    for role in extra_roles:
        out.extend((f"w_{role}_{i}", role, i) for i in range(4))
    return out


def test_select_episodes_exact_quota_and_role_minimums():
    """102 candidates (24/24/54) select exactly 100 episodes with every
    main role >= MIN_EPISODES_PER_ROLE."""
    cands = _candidates(24, 24, 54)
    chosen = select_episodes(cands)
    assert len(chosen) == BANK_SIZE
    per_role = {}
    for _, role, _ in chosen:
        per_role[role] = per_role.get(role, 0) + 1
    for role in CAPABILITY_ROLES:
        assert per_role[role] >= MIN_EPISODES_PER_ROLE, (
            f"role {role} below minimum: {per_role[role]}"
        )
    # deterministic: same input, same selection
    assert select_episodes(cands) == chosen


def test_select_episodes_unknown_role_episodes_counted_in_bank():
    """Extra (unknown) role episodes fill the bank but stay under the
    <= 10 unknown/no-opportunity allowance when small."""
    cands = _candidates(24, 24, 54, extra_roles=("yaw_lateral_recovery",))
    chosen = select_episodes(cands)
    assert len(chosen) == BANK_SIZE
    unknown = [c for c in chosen if c[1] not in CAPABILITY_ROLES]
    assert len(unknown) <= 10


def test_select_episodes_missing_role_rejected():
    """A candidate pool without every main capability role cannot meet
    the Step-2 quotas — hard-fail, never a silent partial bank."""
    with pytest.raises(ValueError, match="lacks capability roles"):
        select_episodes(_candidates(24, 24, 0))


def test_select_episodes_too_few_rejected():
    """Fewer than BANK_SIZE candidates is a hard fail."""
    with pytest.raises(ValueError, match="need"):
        select_episodes(_candidates(20, 20, 20))


def test_select_episodes_role_below_minimum_rejected():
    """Selection that would leave a role below the minimum is a hard
    fail — the bank must be informative per capability."""
    cands = _candidates(10, 45, 45)
    with pytest.raises(ValueError, match="below"):
        select_episodes(cands)


# ═══════════════════════════════════════════════════════════════════
# Step 3 train-leakage check
# ═══════════════════════════════════════════════════════════════════

def test_check_train_leakage_clean():
    """Disjoint world sets pass silently."""
    check_train_leakage(
        ["w_val_0", "w_test_1"], ["w_train_0", "w_train_1"])


def test_check_train_leakage_any_overlap_hard_fails():
    """ANY bank world present in the train manifest is a hard fail —
    the bank is eval-only by construction."""
    with pytest.raises(ValueError, match="train leakage"):
        check_train_leakage(
            ["w_val_0", "w_train_1"], ["w_train_0", "w_train_1"])


# ═══════════════════════════════════════════════════════════════════
# Step 1 aggregate computation
# ═══════════════════════════════════════════════════════════════════

class _Entry:
    """Minimal duck-typed bank entry (role/split/opportunity)."""

    def __init__(self, role, split, opportunity=None):
        self.role = role
        self.split = split
        self.opportunity = opportunity


class _Op:
    """Minimal duck-typed opportunity block."""

    def __init__(self, risk=False, traps=0, stratum="", topo=""):
        self.valid = True
        self.risk_shell_crossing = risk
        self.trap_opportunity_count = traps
        self.support_stratum = stratum
        self.topology_class = topo


def test_compute_aggregate_counts_opportunity_from_blocks():
    """Aggregate opportunity counts come from the frozen blocks, not
    from the world/role names. Dense opportunity (user ruling
    2026-08-24) counts trap AND cluttered topology."""
    entries = [
        _Entry("anticipatory", "rl_val", _Op(risk=True)),
        _Entry("dense_recovery", "rl_test", _Op(traps=1)),
        _Entry("dense_recovery", "rl_val", _Op(topo="cluttered")),
        _Entry("support_expansion", "rl_val", _Op(stratum="inside")),
        _Entry("dense_recovery", "rl_val"),  # no block -> no count
    ]
    agg = compute_aggregate(entries, {"anticipatory": 30,
                                      "full_space": 25,
                                      "dense_recovery": 45})
    assert agg["opportunity"] == {"anticipatory": 1, "full_space": 1,
                                  "dense_recovery": 2}
    assert agg["exposure"] == {"anticipatory": "NOT_YET_MEASURED",
                               "full_space": "NOT_YET_MEASURED",
                               "dense_recovery": "NOT_YET_MEASURED"}
    assert agg["roles"] == {"anticipatory": 1, "dense_recovery": 3,
                            "support_expansion": 1}
    assert agg["worlds"] == {"rl_test": 1, "rl_val": 4}
    assert agg["candidates"] == {"anticipatory": 30, "full_space": 25,
                                 "dense_recovery": 45}


def test_compute_aggregate_candidates_missing_capability_rejected():
    """The candidate counts must cover all three capabilities — a
    missing one would silently under-assert the aggregate."""
    with pytest.raises(ValueError, match="missing capabilities"):
        compute_aggregate([], {"anticipatory": 30, "full_space": 25})


# ═══════════════════════════════════════════════════════════════════
# Step 2 candidate expansion — per-role seed budgets, no yaw episodes
# ═══════════════════════════════════════════════════════════════════

class _Spec:
    """Minimal duck-typed EpisodeTaskSpec (fields the expansion uses)."""

    def __init__(self, world_uid, role):
        self.world_uid = world_uid
        self.role = role
        self.admission_id = f"{world_uid}__{role}"
        self.split = "rl_val"
        self.manifest_entry_hash = "ab" * 32
        self.obstacle_json = f"{world_uid}.world.obstacles.json"
        self.generator_version = "v1"


def test_expand_eval_candidates_per_role_seed_budget():
    """Each spec expands to its role's seed budget; the three main
    roles are covered in manifest order."""
    specs = [_Spec("w_anti_0", "anticipatory"),
             _Spec("w_anti_1", "anticipatory"),
             _Spec("w_dense_0", "dense_recovery")]
    pairs = expand_eval_candidates(specs)
    by_role = {}
    for spec, _ in pairs:
        by_role.setdefault(spec.role, []).append(spec.world_uid)
    assert by_role["anticipatory"] == ["w_anti_0"] * 4 + ["w_anti_1"] * 4
    assert by_role["dense_recovery"] == ["w_dense_0"] * 3
    # manifest order preserved across roles
    assert [s.world_uid for s, _ in pairs] == (
        ["w_anti_0"] * 4 + ["w_anti_1"] * 4 + ["w_dense_0"] * 3)


def test_expand_eval_candidates_excludes_non_capability_roles():
    """Yaw-lateral specs are excluded — their episodes would fall into
    the unknown/no-opportunity allowance (Step 2: <= 10)."""
    specs = [_Spec("w_anti_0", "anticipatory"),
             _Spec("w_yaw_0", "yaw_lateral_recovery")]
    pairs = expand_eval_candidates(specs)
    assert all(s.role != "yaw_lateral_recovery" for s, _ in pairs)
    assert len(pairs) == 4


# ═══════════════════════════════════════════════════════════════════
# freeze + annotate integration — the builder's core composition
# ═══════════════════════════════════════════════════════════════════

class _MockFreezeEnv:
    """The single-slot env duck type (n_envs=1, reset /
    current_episode_config / abort_episode) with a deterministic
    runtime config per (spec, seed)."""

    n_envs = 1

    def __init__(self):
        self._cfg = {}

    def reset(self, specs, seeds):
        spec, seed = specs[0], int(seeds[0])
        self._cfg = {
            "world": spec.world_uid,
            "role": spec.role,
            "seed": seed,
            "start": [-5.0, 0.0, 2.0],
            "goal": [5.0, 0.0, 2.0],
            "start_yaw": 0.0,
            "episode_config_hash": f"cfg_{spec.world_uid}_{seed}",
            "disturb_events": [],
            "start_clearance": 1.0,
            "goal_clearance": 1.0,
            "reachability_pass": True,
            "path_length": 10.0,
            "path_stretch": 1.0,
        }

    def current_episode_config(self):
        return self._cfg

    def abort_episode(self, _idx):
        pass


def test_freeze_and_annotate_applies_opportunity():
    """The composed builder freezes each (spec, seed) through the env
    duck type and annotates the sampled start/goal against the world
    obstacles — every entry carries its geometry opportunity."""
    from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
        OfflineDensity,
    )

    specs = [_Spec("w_anti_0", "anticipatory"),
             _Spec("w_support_0", "support_expansion"),
             _Spec("w_dense_0", "dense_recovery")]
    candidates = expand_eval_candidates(specs)
    cylinder = {"name": "core", "shape": "cylinder", "x": 0.0, "y": 0.0,
                "z": 2.0, "yaw": 0.0, "radius": 0.3, "height": 4.0}
    obstacles = {s.world_uid: [cylinder] for s in specs}
    density = OfflineDensity(counts={(0, 0, 4): 1})
    entries = freeze_and_annotate(
        _MockFreezeEnv(), candidates, obstacles, density,
    )
    assert len(entries) == 11  # 4 + 4 + 3
    for entry in entries:
        assert entry.opportunity is not None
        assert entry.opportunity.valid is True
        # nominal path (-5,0,2) -> (5,0,2) crosses the cylinder
        assert entry.opportunity.risk_shell_crossing is True
        assert entry.episode_config_hash == (
            f"cfg_{entry.world_uid}_{entry.seed}")


def test_freeze_and_annotate_without_density_leaves_support_undecleared():
    """No offline density -> the support group stays undecleared; the
    other groups are still annotated."""
    specs = [_Spec("w_anti_0", "anticipatory")]
    candidates = expand_eval_candidates(specs)
    cylinder = {"name": "core", "shape": "cylinder", "x": 0.0, "y": 0.0,
                "z": 2.0, "yaw": 0.0, "radius": 0.3, "height": 4.0}
    entries = freeze_and_annotate(
        _MockFreezeEnv(), candidates, {"w_anti_0": [cylinder]}, None,
    )
    assert entries[0].opportunity.support_stratum == ""
    assert entries[0].opportunity.risk_shell_crossing is True
