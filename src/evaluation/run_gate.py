"""CI eval gate: run the golden QA suite, diff against the last run, exit 1 on
CRITICAL regression. Mirrors llm-regression-detector's src/runner.py — same
comparator.py, same alerting.py, adapted to RagEvalScore's shape.

Usage:
    python -m src.evaluation.run_gate --chunking-strategy structure_aware [--baseline-run <run_id>] [--slack]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.config import ROOT, settings
from src.evaluation.alerting import send_drift_alert, send_slack_alert
from src.evaluation.comparator import Severity, compare_runs, moving_average_drift
from src.evaluation.evaluator import run_eval_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RUNS_DIR = ROOT / "data" / "eval_runs"


def load_golden_dataset() -> list[dict]:
    """Load golden QA cases from src/evaluation/golden_qa.json.

    Raises:
        FileNotFoundError: if the file is missing.
    """
    path = ROOT / "src" / "evaluation" / "golden_qa.json"
    if not path.exists():
        raise FileNotFoundError(f"Golden QA dataset not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_run_ids() -> list[str]:
    """List saved gate run IDs under data/eval_runs/, oldest first."""
    if not RUNS_DIR.exists():
        return []
    return sorted((p.stem for p in RUNS_DIR.glob("*.json")), key=lambda name: (RUNS_DIR / f"{name}.json").stat().st_mtime)


def load_run(run_id: str) -> dict:
    """Load a saved run by ID.

    Raises:
        FileNotFoundError: if no run with this ID has been saved.
    """
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval run found with ID '{run_id}' (expected {path})")
    return json.loads(path.read_text(encoding="utf-8"))


async def main_async(args: argparse.Namespace) -> int:
    """Run the golden QA suite, save results, diff against a baseline, optionally alert Slack.

    Returns:
        Process exit code: 1 on CRITICAL regression, else 0.
    """
    client = AsyncOpenAI()
    golden_qa = load_golden_dataset()
    run_id = f"{args.chunking_strategy}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"

    summary = await run_eval_suite(golden_qa, client, args.chunking_strategy, run_id)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "run_id": run_id,
        "chunking_strategy": args.chunking_strategy,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aggregate": summary.aggregate,
        "results": [r.model_dump() for r in summary.results],
    }
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    logger.info("Saved eval run %s (%d cases, pass_rate=%.1f%%)", run_id, len(summary.results), summary.aggregate["pass_rate"] * 100)

    prior_run_ids = [r for r in list_run_ids() if r != run_id]
    baseline_run_id = args.baseline_run or (prior_run_ids[-1] if prior_run_ids else None)
    baseline_results = load_run(baseline_run_id)["results"] if baseline_run_id else None

    # comparator.py's category bucketing expects "expected_category"; golden_qa.json calls it "category".
    test_cases_by_id = {tc["id"]: {"expected_category": tc["category"]} for tc in golden_qa}
    comparison = compare_runs(run_id, run_payload["results"], baseline_run_id, baseline_results, test_cases_by_id)

    trend_run_ids = (prior_run_ids + [run_id])[-10:]
    trend_pass_rates = [load_run(rid)["aggregate"]["pass_rate"] for rid in trend_run_ids]

    if args.slack:
        report_url = f"data/eval_runs/{run_id}.json"
        await send_slack_alert(comparison, report_url=report_url)
        drift = moving_average_drift(trend_pass_rates)
        if drift and drift["severity"] != Severity.OK:
            await send_drift_alert(drift, report_url=report_url)

    logger.info(
        "pass_rate=%.1f%% (baseline %.1f%%, delta %+.1f%%), %d regressions, %d improvements, severity=%s",
        comparison.current_pass_rate * 100, comparison.baseline_pass_rate * 100, comparison.pass_rate_delta * 100,
        len(comparison.regressions), len(comparison.improvements), comparison.severity.value,
    )
    return 1 if comparison.severity == Severity.CRITICAL else 0


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the RAG golden QA suite and gate on regressions.")
    parser.add_argument("--chunking-strategy", default=settings.default_chunking_strategy)
    parser.add_argument("--baseline-run", default=None, help="Run ID to compare against (default: most recent prior run)")
    parser.add_argument("--slack", action="store_true", help="Send Slack alert after the run")
    args = parser.parse_args()

    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
