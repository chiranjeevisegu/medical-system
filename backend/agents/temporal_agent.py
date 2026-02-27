from __future__ import annotations

import json
import re
from typing import Any

from backend.model_loader import generate_reasoning


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


class TemporalProgressionAgent:
    """Trend detector across current report and optional previous report."""

    def run(self, current_report: str, previous_report: str | None = None, lab_context: list[dict[str, Any]] | None = None) -> dict:
        previous = previous_report.strip() if previous_report else ""
        lab_blob = json.dumps(lab_context or [], ensure_ascii=False)

        prompt = (
            "Analyze temporal progression between current and previous report.\n"
            "If previous report is unavailable, infer trend only from current report and labs.\n"
            "Return strict JSON with keys: trend, change_summary, evidence.\n"
            "trend must be one of: Improving, Worsening, Stable, Unclear.\n"
            f"CURRENT_REPORT:\n{current_report}\n"
            f"PREVIOUS_REPORT:\n{previous}\n"
            f"LAB_CONTEXT:\n{lab_blob}"
        )
        parsed = _parse_json_response(generate_reasoning(prompt))
        trend = str(parsed.get("trend", "Unclear")).strip().title()
        if trend not in {"Improving", "Worsening", "Stable", "Unclear"}:
            trend = "Unclear"

        return {
            "trend": trend,
            "change_summary": str(parsed.get("change_summary", "")).strip(),
            "evidence": [str(x) for x in parsed.get("evidence", []) if str(x).strip()],
        }
