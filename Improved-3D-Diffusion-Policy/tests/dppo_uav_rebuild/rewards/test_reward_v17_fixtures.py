"""Fixture manifest must pass the real geometry/reward calibration consumers."""

from diffusion_policy_3d.env.reward_v17 import resolve_config
from diffusion_policy_3d.env.reward_v17_support.calibration import (
    anchor_scale, deterministic_signal_checks, replay_episodes, select_height_weights,
)
from diffusion_policy_3d.env.reward_v17_support.config import DEFAULTS
from diffusion_policy_3d.env.reward_v17_support.density import build_density
from diffusion_policy_3d.env.reward_v17_support.fixture_manifest import make_fixture_manifest
from scripts.prepare_reward_v17 import check_ordering_manifest, measure_anchors, replay_contract_checks


def fixture_density():
    return build_density([
        dict(episode_id="demo", split="bc_demo", source="frozen_ego", start=[0, 0, 1],
             goal=[4, 0, 1], ground_z=0, initial_yaw=0, positions=[[0, 0, 1], [1, 0, 1]])
    ], dataset_sha256="d" * 64)


def test_fixture_manifest_passes_real_replay_and_resolves_finite_search():
    manifest = make_fixture_manifest("generator.py", "a" * 64)
    pairs = check_ordering_manifest(manifest)
    empirical, fixtures = measure_anchors(manifest["episodes"], DEFAULTS)
    scale = anchor_scale(empirical, fixtures)
    assert scale["status"] == "resolved" and scale["calibration_source"] == "fixture"
    cfg = resolve_config(dict(gamma_pbrs=.99, d_ref=scale["d_ref"], s_f=scale["s_f"],
                              w_low=2.5, w_high=1., w_climb=1., sigma_z=.3))
    density = fixture_density()
    features, _, records = replay_episodes(manifest["episodes"], cfg, density)
    checks = deterministic_signal_checks(cfg, density)
    checks.update(replay_contract_checks(manifest, records, density))
    search = select_height_weights(features, pairs, checks)
    assert all(checks.values())
    assert search["status"] == "resolved"
    assert search["selection_objective"] == "discounted_return"
    assert "failed_undiscounted_orderings" in search["selected"]


def test_manifest_covers_each_required_outcome_and_keeps_tasks_paired():
    manifest = make_fixture_manifest("generator.py", "a" * 64)
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    assert {e["expected_done_reason"] for e in episodes.values()} >= {"reached", "collision", "time_limit"}
    assert len(manifest["ordering_categories"]) == 10
    assert len(episodes["wall_timeout"]["steps"]) == 60
    for pairs in manifest["ordering_categories"].values():
        for better, worse in pairs:
            for key in ("start", "goal", "obstacles", "geometry_context"):
                assert episodes[better][key] == episodes[worse][key]
