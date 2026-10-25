"""Evidence labels that prevent surrogate results from becoming formal claims."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionEvidence:
    paired_metrics: dict
    independent_training_seeds: int
    planned_training_budget_completed: bool
    long_run_stability_passed: bool
    real_e5_parity_passed: bool
    gazebo_px4_evaluated: bool


@dataclass(frozen=True)
class AdmissionDecision:
    positive_lightweight_signal: bool
    formal_rl_training_claim: bool
    gazebo_capability_claim: bool
    blockers: tuple


def decide_admission(evidence: AdmissionEvidence):
    metrics = evidence.paired_metrics
    positive = (
        metrics["return"].lower > 0.0
        and metrics["success"].mean > 0.0
        and metrics["collision"].upper <= 0.0
        and metrics["onset_distance"].mean > 0.0
        and metrics["support_cells"].mean > 0.0
    )
    blockers = []
    if not evidence.real_e5_parity_passed:
        blockers.append("REAL_E5_PARITY")
    if evidence.independent_training_seeds < 3:
        blockers.append("B7_MULTI_SEED")
    if not evidence.planned_training_budget_completed:
        blockers.append("PLANNED_BUDGET")
    if not evidence.long_run_stability_passed:
        blockers.append("B8_LONG_RUN")
    if not positive:
        blockers.append("POSITIVE_PAIRED_BENEFIT")
    formal = not blockers
    return AdmissionDecision(
        positive_lightweight_signal=positive,
        formal_rl_training_claim=formal,
        gazebo_capability_claim=formal and evidence.gazebo_px4_evaluated,
        blockers=tuple(blockers),
    )

