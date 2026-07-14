"""Generate a mock chunking-strategy comparison — no OpenAI calls, no cost.

For portfolio/demo purposes: synthesizes plausible per-case RagEvalScores for
each of the 55 golden QA cases under each chunking strategy, modeled with the
tradeoffs the pipeline is designed to demonstrate:
  - structure_aware: best overall — markdown headers line up with how the
    corpus is actually organized.
  - semantic: close behind, strongest specifically on multi_hop questions
    (coherent topic boundaries), costs the most (per-sentence embedding calls).
  - fixed_size: weakest, especially on technical/adversarial cases where a
    token-count cut can split an error code or a qualifying clause from its
    context.

Usage: python scripts/generate_demo_data.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import RagEvalScore, aggregate_metrics  # noqa: E402

# Per-strategy, per-category base pass probability. Tuned so structure_aware
# wins overall, semantic wins on multi_hop, fixed_size lags on technical/adversarial.
BASE_RATES = {
    "fixed_size": {"lookup": 0.85, "multi_hop": 0.55, "technical": 0.60, "unanswerable": 0.70, "ambiguous": 0.55, "adversarial": 0.40},
    "structure_aware": {"lookup": 0.95, "multi_hop": 0.75, "technical": 0.85, "unanswerable": 0.85, "ambiguous": 0.65, "adversarial": 0.60},
    "semantic": {"lookup": 0.90, "multi_hop": 0.82, "technical": 0.78, "unanswerable": 0.80, "ambiguous": 0.68, "adversarial": 0.62},
}
STRATEGY_COST_PER_CASE = {"fixed_size": 0.0009, "structure_aware": 0.0011, "semantic": 0.0016}


def mock_case_score(test_case: dict, passed: bool, rng: random.Random) -> RagEvalScore:
    correctness = rng.uniform(0.75, 1.0) if passed else rng.uniform(0.1, 0.5)
    faithfulness = rng.uniform(0.8, 1.0) if passed else rng.uniform(0.2, 0.6)
    retrieval_relevance = rng.uniform(0.7, 1.0) if passed else rng.uniform(0.0, 0.5)
    citation_accuracy = rng.uniform(0.8, 1.0) if passed else rng.uniform(0.3, 0.7)

    has_answer = test_case["has_answer_in_corpus"]
    unanswerable_detection = passed if test_case["category"] == "unanswerable" else rng.random() < 0.95

    return RagEvalScore(
        test_case_id=test_case["id"],
        answer_correctness=correctness,
        faithfulness=faithfulness,
        retrieval_relevance=retrieval_relevance if has_answer else 1.0,
        citation_accuracy=citation_accuracy,
        unanswerable_detection=unanswerable_detection,
        passed=passed,
    )


def _assign_pass_fail(cases: list[dict], strategy: str, rng: random.Random) -> dict[str, bool]:
    """Deterministic per-category pass quota (round(base_rate * n)) rather than
    independent per-case coin-flips — at ~9 cases/category, binomial sampling
    noise was enough to bury the intended strategy differences entirely."""
    by_category: dict[str, list[dict]] = {}
    for tc in cases:
        by_category.setdefault(tc["category"], []).append(tc)

    passed_by_id: dict[str, bool] = {}
    for category, group in by_category.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        num_passing = round(BASE_RATES[strategy][category] * len(shuffled))
        for i, tc in enumerate(shuffled):
            passed_by_id[tc["id"]] = i < num_passing
    return passed_by_id


def main() -> None:
    golden_qa = json.loads((ROOT / "src" / "evaluation" / "golden_qa.json").read_text(encoding="utf-8"))

    report_rows = []
    for strategy, seed in (("fixed_size", 1), ("structure_aware", 2), ("semantic", 3)):
        rng = random.Random(seed)
        passed_by_id = _assign_pass_fail(golden_qa, strategy, rng)
        results = [mock_case_score(tc, passed_by_id[tc["id"]], rng) for tc in golden_qa]
        aggregate = aggregate_metrics(results)
        cost = round(len(golden_qa) * STRATEGY_COST_PER_CASE[strategy], 4)

        report_rows.append(
            {
                "strategy": strategy,
                "chunks_stored": {"fixed_size": 17, "structure_aware": 59, "semantic": 48}[strategy],
                **aggregate,
                "cost_usd": cost,
            }
        )

    output = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "strategies": report_rows, "note": "mock data — no OpenAI calls were made; see scripts/generate_demo_data.py"}
    output_path = ROOT / "data" / "chunking_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for row in report_rows:
        print(f"{row['strategy']:>16}: pass_rate={row['pass_rate']:.1%} correctness={row['answer_correctness']:.1%} cost=${row['cost_usd']}")
    print(f"written to {output_path}")


if __name__ == "__main__":
    main()
