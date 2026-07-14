"""Slack webhook alerting for eval run results and slow-drift detection."""

from __future__ import annotations

import logging
import os

import httpx

from src.evaluation.comparator import ComparisonResult, Severity

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {Severity.OK: "✅", Severity.WARNING: "⚠️", Severity.CRITICAL: "🔴"}


async def _post_to_slack(text: str) -> None:
    """POST a message to SLACK_WEBHOOK_URL. No-ops with a warning if the env var is unset.

    Raises:
        httpx.HTTPStatusError: if Slack rejects the webhook request.
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set, skipping Slack alert")
        return

    async with httpx.AsyncClient() as client:
        response = await client.post(webhook_url, json={"text": text})
        response.raise_for_status()
    logger.info("Slack alert sent")


async def send_slack_alert(comparison: ComparisonResult, report_url: str) -> None:
    """Post a run-vs-baseline regression summary to Slack.

    Args:
        comparison: Output of comparator.compare_runs.
        report_url: Link to the run's HTML report, included in the message.
    """
    emoji = SEVERITY_EMOJI[comparison.severity]
    headline = (
        f"{emoji} *{len(comparison.regressions)} regressions detected* — "
        f"accuracy {comparison.baseline_pass_rate:.0%}→{comparison.current_pass_rate:.0%}"
    )
    text = (
        f"{headline}\n"
        f"Run: `{comparison.current_run_id}` vs baseline `{comparison.baseline_run_id or 'none'}`\n"
        f"Improvements: {len(comparison.improvements)}\n"
        f"<{report_url}|View full report>"
    )
    await _post_to_slack(text)


async def send_drift_alert(drift: dict, report_url: str) -> None:
    """Post a slow-drift alert to Slack (latest run vs. trailing moving average).

    Args:
        drift: Output of comparator.moving_average_drift.
        report_url: Link to the run's HTML report, included in the message.
    """
    emoji = SEVERITY_EMOJI[drift["severity"]]
    text = (
        f"{emoji} *Slow drift detected* — latest pass rate {drift['latest']:.1%} vs "
        f"7-run moving average {drift['moving_average']:.1%} ({drift['delta']:+.1%})\n"
        f"<{report_url}|View full report>"
    )
    await _post_to_slack(text)
