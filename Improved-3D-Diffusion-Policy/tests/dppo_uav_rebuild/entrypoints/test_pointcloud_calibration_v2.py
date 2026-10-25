import pytest

from scripts.calibrate_routea_v17_pointcloud import calibrate, load_transition_rows


def test_calibration_is_bc_only_and_does_not_turn_sample_max_into_hard_gate():
    rows = [
        {"policy_label": "bc", "pointcloud_no_ray_hit_fraction": index / 4096,
         "pointcloud_valid_hit_rays": 4096 - index, "pointcloud_total_rays": 4096}
        for index in range(1024)
    ]
    result = calibrate(rows)
    assert result["max_observed_no_ray_hit_fraction"] == pytest.approx(1023 / 4096)
    assert result["max_no_ray_hit_fraction"] == pytest.approx(4095 / 4096)
    assert result["partial_no_ray_action"] == "monitor_only"
    assert result["content_lower_bound"] > 0.99


def test_calibration_rejects_rl_and_all_no_ray():
    rows = [{"policy_label": "rl", "pointcloud_no_ray_hit_fraction": 0.1,
             "pointcloud_valid_hit_rays": 1, "pointcloud_total_rays": 10}] * 1024
    with pytest.raises(ValueError, match="BC records only"):
        calibrate(rows)
    rows = [{"policy_label": "bc", "pointcloud_no_ray_hit_fraction": 1.0,
             "pointcloud_valid_hit_rays": 0, "pointcloud_total_rays": 10}] * 1024
    with pytest.raises(ValueError, match="invalid"):
        calibrate(rows)


def test_calibration_accepts_explicit_verified_no_return_frames():
    rows = [{
        "policy_label": "bc",
        "pointcloud_no_ray_hit_fraction": 1.0,
        "pointcloud_valid_hit_rays": 0,
        "pointcloud_total_rays": 10,
        "pointcloud_used_far_plane_fallback": True,
    }] * 1024
    result = calibrate(rows)
    assert result["verified_no_return_frame_count"] == 1024
    assert result["all_no_ray_policy"] == "accept_verified_max_range_censoring"


def test_calibration_loader_rejects_duplicate_keys_and_oversized_lines(tmp_path):
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"policy_label":"bc","policy_label":"rl","pointcloud_no_ray_hit_fraction":0.2,"pointcloud_valid_hit_rays":8,"pointcloud_total_rays":10}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_transition_rows(duplicate)

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_text(
        '{"padding":"' + ("x" * (1024 * 1024 + 1)) + '"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line exceeds"):
        load_transition_rows(oversized)
