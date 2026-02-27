from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.baseline import BaselineSummarizer
from backend.config import get_settings
from backend.evaluation import Evaluator
from backend.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run medical-agentic-system in CLI mode and print outputs."
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=5,
        help="Number of cases to analyze (default: 5).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save analyzed outputs as JSON.",
    )
    parser.add_argument(
        "--eval-cases",
        type=int,
        default=20,
        help="Number of cases to include in evaluation metrics (default: 20).",
    )
    args = parser.parse_args()

    settings = get_settings()
    orchestrator = Orchestrator(settings)
    baseline = BaselineSummarizer()
    evaluator = Evaluator(orchestrator, baseline, settings.results_path)

    case_ids = orchestrator.list_case_ids()[: max(1, args.cases)]
    analyzed_outputs: list[dict] = []

    print(f"Running analysis for {len(case_ids)} case(s)...")
    for idx, case_id in enumerate(case_ids, start=1):
        try:
            result = orchestrator.analyze_case(case_id)
        except Exception as exc:
            print("\n" + "=" * 90)
            print(f"[{idx}/{len(case_ids)}] CASE: {case_id}")
            print("ERROR:", str(exc))
            continue
        analyzed_outputs.append({"case_id": case_id, "result": result})

        clinical = result.get("clinical_summary", {})
        risk = result.get("risk", {})
        explanation = result.get("explanation", {})
        recommendation = result.get("recommendation", {})
        justification = result.get("justification", {})
        verification = result.get("verification", {})

        print("\n" + "=" * 90)
        print(f"[{idx}/{len(case_ids)}] CASE: {case_id}")
        print(
            "CLASS:",
            clinical.get("disease_category", "N/A"),
            "| CONF:",
            clinical.get("disease_confidence", "N/A"),
        )
        print("RISK:", risk.get("risk_level", "N/A"))
        print("EXPLANATION:", (explanation.get("simple_explanation", "") or "")[:500])
        print("TAKEAWAY:", (explanation.get("plain_language_takeaway", "") or "")[:300])
        print("SAFE_NEXT_STEPS:", recommendation.get("safe_next_steps", []))
        print("URGENT_SIGNS:", recommendation.get("urgent_attention_signs", []))
        print("JUSTIFICATION_CONF:", justification.get("confidence", "N/A"))
        print("VERIFICATION_SCORE:", verification.get("consistency_score", "N/A"))

    print("\n" + "=" * 90)
    print("Running evaluation...")
    eval_case_ids = orchestrator.list_case_ids()[: max(1, args.eval_cases)]
    metrics = evaluator.evaluate_all(case_ids=eval_case_ids)
    print(json.dumps(metrics, indent=2))
    print(f"\nResults file written: {settings.results_path}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(analyzed_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Detailed case outputs saved: {out_path}")


if __name__ == "__main__":
    main()
