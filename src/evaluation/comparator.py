"""Diff logic between eval runs: pass-rate deltas, regressions, and drift detection."""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WARNING_THRESHOLD = 0.03
CRITICAL_THRESHOLD = 0.08


class Severity(str, Enum):
    """Regression severity, driven by how far pass rate dropped vs. baseline/trend."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class ComparisonResult(BaseModel):
    """Diff between a current eval run and a baseline run."""

    baseline_run_id: str | None = Field(..., description="Run ID compared against, or None if there was no baseline.")
    current_run_id: str = Field(..., description="Run ID being evaluated.")
    baseline_pass_rate: float = Field(..., ge=0.0, le=1.0, description="Baseline run's overall pass rate.")
    current_pass_rate: float = Field(..., ge=0.0, le=1.0, description="Current run's overall pass rate.")
    pass_rate_delta: float = Field(..., description="current_pass_rate - baseline_pass_rate.")
    category_deltas: dict[str, float] = Field(..., description="Per-category pass rate delta, current minus baseline.")
    regressions: list[str] = Field(..., description="Test case IDs that passed on baseline but failed on current.")
    improvements: list[str] = Field(..., description="Test case IDs that failed on baseline but passed on current.")
    severity: Severity = Field(..., description="OK / WARNING / CRITICAL based on the pass-rate drop thresholds.")


def _pass_rate(scores: list[dict]) -> float:
    """Fraction of scores with passed=True. Returns 0.0 for an empty list."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s["passed"]) / len(scores)


def _category_pass_rates(scores: list[dict], test_cases_by_id: dict[str, dict]) -> dict[str, float]:
    """Group scores by their test case's expected_category and compute a pass rate per group.

    Raises:
        KeyError: if a score references a test_case_id not present in test_cases_by_id.
    """
    by_category: dict[str, list[dict]] = {}
    for s in scores:
        tc_id = s["test_case_id"]
        if tc_id not in test_cases_by_id:
            raise KeyError(f"Score references unknown test_case_id '{tc_id}' not found in golden dataset")
        category = test_cases_by_id[tc_id]["expected_category"]
        by_category.setdefault(category, []).append(s)
    return {cat: _pass_rate(items) for cat, items in by_category.items()}


def _severity_for_drop(drop: float) -> Severity:
    """Classify a pass-rate drop (positive = got worse) against the warning/critical thresholds."""
    if drop >= CRITICAL_THRESHOLD:
        return Severity.CRITICAL
    if drop >= WARNING_THRESHOLD:
        return Severity.WARNING
    return Severity.OK


def compare_runs(
    current_run_id: str,
    current_scores: list[dict],
    baseline_run_id: str | None,
    baseline_scores: list[dict] | None,
    test_cases_by_id: dict[str, dict],
) -> ComparisonResult:
    """Diff a current eval run against a baseline run.

    Args:
        current_run_id: ID of the run being evaluated.
        current_scores: List of per-case score dicts (each with "test_case_id", "passed").
        baseline_run_id: ID of the baseline run, or None if there isn't one yet.
        baseline_scores: Baseline run's score dicts, or None if there isn't a baseline.
        test_cases_by_id: Golden dataset entries keyed by test case ID, used to bucket
            per-category pass rates.

    Returns:
        A ComparisonResult with pass-rate deltas, regressed/improved case IDs, and severity.

    Raises:
        ValueError: if current_scores is empty (nothing to compare).
    """
    if not current_scores:
        raise ValueError("current_scores must not be empty")

    current_pass_rate = _pass_rate(current_scores)
    baseline_pass_rate = _pass_rate(baseline_scores) if baseline_scores else current_pass_rate
    pass_rate_delta = current_pass_rate - baseline_pass_rate

    current_by_id = {s["test_case_id"]: s["passed"] for s in current_scores}
    baseline_by_id = {s["test_case_id"]: s["passed"] for s in (baseline_scores or [])}

    if baseline_scores:
        regressions = [
            tc_id for tc_id, passed in current_by_id.items() if baseline_by_id.get(tc_id, True) and not passed
        ]
        improvements = [
            tc_id for tc_id, passed in current_by_id.items() if not baseline_by_id.get(tc_id, False) and passed
        ]
    else:
        # No baseline to diff against — nothing has "regressed" or "improved" yet.
        regressions = []
        improvements = []

    current_cat_rates = _category_pass_rates(current_scores, test_cases_by_id)
    baseline_cat_rates = (
        _category_pass_rates(baseline_scores, test_cases_by_id) if baseline_scores else current_cat_rates
    )
    category_deltas = {
        cat: current_cat_rates.get(cat, 0.0) - baseline_cat_rates.get(cat, 0.0)
        for cat in set(current_cat_rates) | set(baseline_cat_rates)
    }

    severity = _severity_for_drop(-pass_rate_delta)
    if severity != Severity.OK:
        logger.warning(
            "Eval run %s vs baseline %s: severity=%s, pass_rate_delta=%.1f%%, %d regressions",
            current_run_id, baseline_run_id, severity.value, pass_rate_delta * 100, len(regressions),
        )

    return ComparisonResult(
        baseline_run_id=baseline_run_id,
        current_run_id=current_run_id,
        baseline_pass_rate=baseline_pass_rate,
        current_pass_rate=current_pass_rate,
        pass_rate_delta=pass_rate_delta,
        category_deltas=category_deltas,
        regressions=regressions,
        improvements=improvements,
        severity=severity,
    )


def moving_average_drift(pass_rates: list[float], window: int = 7) -> dict | None:
    """Detect slow drift by comparing the latest run against a trailing moving average.

    Unlike compare_runs (which only looks at the immediately prior run), this catches
    gradual degradation across several runs that each individually stay under threshold.

    Args:
        pass_rates: Pass rates in chronological order, oldest first.
        window: Number of prior runs to average over (default 7).

    Returns:
        None if there isn't enough history (fewer than 2 runs) to compare against.
        Otherwise a dict with moving_average, latest, delta, and severity.
    """
    if len(pass_rates) < 2:
        return None
    trailing = pass_rates[-(window + 1) : -1] or pass_rates[:-1]
    avg = sum(trailing) / len(trailing)
    latest = pass_rates[-1]
    delta = latest - avg
    severity = _severity_for_drop(-delta)
    return {"moving_average": avg, "latest": latest, "delta": delta, "severity": severity}
