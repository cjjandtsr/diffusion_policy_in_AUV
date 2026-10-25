import pytest

from scripts.dppo_credit_support.credit_probe import assess_sampled_k_gate


def test_per_role_gate_rejects_hidden_amplification_and_attenuation():
    reference = {
        "anticipatory": [1.0, 0.0],
        "dense_recovery": [0.0, 1.0],
    }
    schedules = [
        {
            "anticipatory": [2.1, 0.0],
            "dense_recovery": [0.0, 0.49],
        }
        for _ in range(100)
    ]
    report = assess_sampled_k_gate(reference, schedules)
    assert report["decision"] == "BLOCK"
    assert "ROLE_NORM_RATIO" in report["failure_codes"]


def test_zero_reference_norm_is_undefined_and_blocks():
    report = assess_sampled_k_gate(
        {"anticipatory": [0.0, 0.0]},
        [{"anticipatory": [0.0, 0.0]} for _ in range(100)],
    )
    assert report["decision"] == "BLOCK"
    assert "UNDEFINED_ROLE_NORM_RATIO" in report["failure_codes"]


def test_exact_schedules_pass_without_inverse_k_bias():
    reference = {
        "anticipatory": [1.0, 2.0],
        "dense_recovery": [-2.0, 1.0],
    }
    report = assess_sampled_k_gate(reference, [reference] * 100)
    assert report["decision"] == "PASS"
    assert report["overall"]["mean_gradient_cosine"] == pytest.approx(1.0)
    assert report["overall"]["relative_norm_bias"] == pytest.approx(0.0)
