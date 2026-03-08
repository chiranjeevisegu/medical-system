from __future__ import annotations

import json
import re

from backend.model_loader import generate_reasoning
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_json_response(text: str) -> dict:
    cleaned = (text or "").strip()
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


def _clean_item(text: str, max_chars: int = 200) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return cleaned


def _fallback_verification(
    clinical_summary: dict,
    risk: dict,
    explanation: dict,
    recommendation: dict,
    justification: dict,
) -> dict:
    contradictions: list[str] = []
    placeholder_tokens = ("item1", "item2", "item3", "placeholder")

    risk_level = str((risk or {}).get("risk_level", "")).strip().title()
    if risk_level not in {"Low", "Medium", "High"}:
        contradictions.append("Risk level missing or malformed.")

    explanation_text = str((explanation or {}).get("simple_explanation", "")).strip()
    take_away = str((explanation or {}).get("plain_language_takeaway", "")).strip()
    rec_count = len((recommendation or {}).get("safe_next_steps", []) or []) + len(
        (recommendation or {}).get("urgent_attention_signs", []) or []
    )
    rationale_count = len((justification or {}).get("rationale", []) or [])

    if not explanation_text or not take_away:
        contradictions.append("Explainability output incomplete.")
    if rec_count == 0:
        contradictions.append("Recommendation output missing actionable items.")
    if rationale_count == 0:
        contradictions.append("Justification output missing rationale.")

    recommendation_text = json.dumps(recommendation or {}, ensure_ascii=False).lower()
    justification_text = json.dumps(justification or {}, ensure_ascii=False).lower()
    if any(tok in recommendation_text for tok in placeholder_tokens):
        contradictions.append("Recommendation contains placeholder content.")
    if any(tok in justification_text for tok in placeholder_tokens):
        contradictions.append("Justification contains placeholder content.")

    clinical_items = 0
    for key in ("diagnosis", "abnormal_findings", "key_observations"):
        values = (clinical_summary or {}).get(key, [])
        if isinstance(values, list):
            clinical_items += len([v for v in values if str(v).strip()])
    if clinical_items == 0:
        contradictions.append("Clinical extraction appears empty.")

    score = 0.9
    score -= min(0.35, 0.1 * len(contradictions))
    score = max(0.4, min(0.95, score))

    return {
        "contradictions": contradictions,
        "consistency_score": round(score, 3),
        "safety_notes": [
            "Consistency score is heuristic when strict JSON verification is unavailable.",
            "All outputs remain non-diagnostic and should be clinician-reviewed.",
        ],
    }


class VerificationAgent:
    """FLAN-T5 cross-agent consistency checker."""

    def run(
        self,
        clinical_summary: dict,
        risk: dict,
        explanation: dict,
        recommendation: dict,
        justification: dict,
    ) -> dict:
        prompt = (
            "Verify consistency across all agent outputs and detect contradictions.\n"
            "Return strict JSON ONLY with these keys:\n"
            "  contradictions: list of string (empty list if none),\n"
            "  consistency_score: float between 0 and 1 (higher = more consistent),\n"
            "  safety_notes: list of string.\n"
            "Do NOT include text before or after the JSON.\n"
            f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}\n"
            f"RISK:\n{json.dumps(risk, ensure_ascii=False)}\n"
            f"EXPLANATION:\n{json.dumps(explanation, ensure_ascii=False)}\n"
            f"RECOMMENDATION:\n{json.dumps(recommendation, ensure_ascii=False)}\n"
            f"JUSTIFICATION:\n{json.dumps(justification, ensure_ascii=False)}"
        )
        logger.debug("VerificationAgent: attempt 1")
        try:
            parsed = _parse_json_response(generate_reasoning(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.error("VerificationAgent generate_reasoning failed (attempt 1): %s", exc)
            parsed = {}

        try:
            score = float(parsed.get("consistency_score", 0.5))
        except (TypeError, ValueError):
            score = 0.5

        result = {
            "contradictions": [str(x) for x in parsed.get("contradictions", []) if str(x).strip()],
            "consistency_score": max(0.0, min(1.0, score)),
            "safety_notes": [str(x) for x in parsed.get("safety_notes", []) if str(x).strip()],
        }

        # If LLM returned default empty response, use heuristic fallback
        if result["consistency_score"] == 0.5 and not result["contradictions"] and not result["safety_notes"]:
            logger.debug("VerificationAgent: using heuristic fallback")
            return _fallback_verification(
                clinical_summary=clinical_summary,
                risk=risk,
                explanation=explanation,
                recommendation=recommendation,
                justification=justification,
            )

        # Clean text items
        result["contradictions"] = [_clean_item(x, 200) for x in result["contradictions"][:5]]
        result["safety_notes"] = [_clean_item(x, 200) for x in result["safety_notes"][:5]]
        if not result["safety_notes"]:
            result["safety_notes"] = ["All outputs are non-diagnostic and require clinician review."]
        result["consistency_score"] = round(result["consistency_score"], 3)
        return result
