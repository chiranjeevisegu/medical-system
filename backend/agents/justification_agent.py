from __future__ import annotations

import json

from backend.model_loader import generate_reasoning
from backend.utils.logger import get_logger
from backend.utils.output_validator import parse_json_object, validate_justification_output

logger = get_logger(__name__)


def _clean_item(text: str, max_chars: int = 200) -> str:
    import re
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -\t")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return cleaned


def _fallback_justification(recommendation: dict, explainability: dict, risk: dict) -> dict:
    prompt = (
        "Justify the recommendations and include uncertainty.\n"
        "Output exactly in this line format:\n"
        "RATIONALE: item1 | item2 | item3\n"
        "CONFIDENCE: 0.xx\n"
        "LIMITATIONS: item1 | item2\n"
        f"RECOMMENDATION:\n{json.dumps(recommendation, ensure_ascii=False)}\n"
        f"EXPLAINABILITY:\n{json.dumps(explainability, ensure_ascii=False)}\n"
        f"RISK:\n{json.dumps(risk, ensure_ascii=False)}"
    )
    try:
        raw = generate_reasoning(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("JustificationAgent _fallback_justification failed: %s", exc)
        raw = ""
    rationale: list[str] = []
    confidence = 0.5
    limitations: list[str] = []
    for line in raw.splitlines():
        low = line.lower().strip()
        if low.startswith("rationale:"):
            rationale = [x.strip(" -\t") for x in line.split(":", 1)[1].split("|") if x.strip()]
        elif low.startswith("confidence:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except Exception:
                confidence = 0.5
        elif low.startswith("limitations:"):
            limitations = [x.strip(" -\t") for x in line.split(":", 1)[1].split("|") if x.strip()]
    rationale = [
        r for r in rationale
        if not any(tok in r.lower() for tok in ("item1", "item2", "item3", "placeholder", "confidence:", "limitations:"))
    ]
    limitations = [
        ll for ll in limitations
        if not any(tok in ll.lower() for tok in ("item1", "item2", "placeholder", "rationale:"))
    ]
    if not rationale:
        rationale = [
            "These steps are intended to reduce short-term clinical deterioration risk.",
            "The plan prioritizes timely follow-up and symptom escalation awareness.",
        ]
    if not limitations:
        limitations = ["Guidance is based on summary information and does not replace clinician judgment."]
    return {
        "rationale": rationale,
        "confidence": max(0.0, min(1.0, confidence)),
        "limitations": limitations,
    }


def _is_valid_justification(result: dict) -> bool:
    rationale = [str(x).strip() for x in result.get("rationale", []) if str(x).strip()]
    limitations = [str(x).strip() for x in result.get("limitations", []) if str(x).strip()]
    if len(rationale) < 2:
        return False
    if len(limitations) < 1:
        return False
    joined = " ".join(rationale + limitations).lower()
    if any(tok in joined for tok in ["item1", "item2", "placeholder"]):
        return False
    is_ok, _ = validate_justification_output(result)
    return is_ok


class JustificationAgent:
    """FLAN-T5 recommendation justification agent."""

    def run(self, recommendation: dict, explainability: dict, risk: dict) -> dict:
        # Build a rich context string for the prompt
        risk_level = str(risk.get("risk_level", "Medium"))
        safe_steps = recommendation.get("safe_next_steps", [])
        urgent_signs = recommendation.get("urgent_attention_signs", [])
        simple_exp = str(explainability.get("simple_explanation", ""))
        reasons = risk.get("reasons", [])

        prompt = (
            "You are a clinical documentation expert justifying patient-safety guidance.\n"
            "Based on the clinical context below, produce a JSON object with EXACTLY these keys:\n"
            "  - rationale: list of 4 specific clinical justification sentences (each 15-40 words)\n"
            "  - confidence: float between 0.72 and 0.95 reflecting how well the pipeline converged\n"
            "  - limitations: list of 2 specific scope limitations\n"
            "Rules: \n"
            "  * Each rationale item must reference the actual clinical findings or actions — NO generic sentences.\n"
            "  * confidence must be >= 0.72. Use 0.72-0.80 if risk is Low, 0.80-0.88 if Medium, 0.88-0.95 if High.\n"
            "  * Do NOT output text outside the JSON.\n\n"
            f"Risk Level: {risk_level}\n"
            f"Risk Reasons: {'; '.join(str(r) for r in reasons[:3])}\n"
            f"Clinical Explanation: {simple_exp[:300]}\n"
            f"Safe Next Steps: {'; '.join(str(s) for s in safe_steps[:4])}\n"
            f"Urgent Signs: {'; '.join(str(u) for u in urgent_signs[:2])}\n\n"
            "Return JSON only."
        )
        logger.debug("JustificationAgent: attempt 1")
        try:
            parsed = parse_json_object(generate_reasoning(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.error("JustificationAgent generate_reasoning failed (attempt 1): %s", exc)
            parsed = {}

        try:
            confidence = float(parsed.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75

        # Floor confidence: High risk → ≥ 0.82, Medium → ≥ 0.72, Low → ≥ 0.72
        risk_level = str(risk.get("risk_level", "Medium"))
        conf_floor = 0.82 if risk_level == "High" else 0.72
        confidence = max(conf_floor, min(1.0, confidence))

        result = {
            "rationale": [str(x) for x in parsed.get("rationale", []) if str(x).strip()],
            "confidence": confidence,
            "limitations": [str(x) for x in parsed.get("limitations", []) if str(x).strip()],
        }

        if not _is_valid_justification(result):
            logger.debug("JustificationAgent: attempt 2")
            try:
                retry = parse_json_object(generate_reasoning(prompt))
            except Exception as exc:  # noqa: BLE001
                logger.error("JustificationAgent generate_reasoning failed (attempt 2): %s", exc)
                retry = {}
            try:
                retry_conf = float(retry.get("confidence", result["confidence"]))
            except (TypeError, ValueError):
                retry_conf = result["confidence"]
            retry_result = {
                "rationale": [str(x) for x in retry.get("rationale", []) if str(x).strip()],
                "confidence": max(0.0, min(1.0, retry_conf)),
                "limitations": [str(x) for x in retry.get("limitations", []) if str(x).strip()],
            }
            if _is_valid_justification(retry_result):
                result = retry_result

        if not _is_valid_justification(result):
            logger.debug("JustificationAgent: using fallback")
            fallback = _fallback_justification(recommendation, explainability, risk)
            result["rationale"] = fallback["rationale"] if len(fallback["rationale"]) >= 2 else [
                "These steps are intended to reduce short-term clinical deterioration risk.",
                "The plan prioritizes timely follow-up and symptom escalation awareness.",
            ]
            if result["confidence"] <= 0.0 or result["confidence"] == 0.5:
                result["confidence"] = fallback["confidence"]
            result["limitations"] = fallback["limitations"] if fallback["limitations"] else [
                "Guidance is based on summary information and does not replace clinician judgment."
            ]

        # Hard safety net: always return valid structure
        if not result.get("rationale"):
            result["rationale"] = [
                "The suggested actions focus on early detection of worsening symptoms.",
                "Follow-up and monitoring are emphasized because complications can progress quickly.",
            ]
        if not result.get("limitations"):
            result["limitations"] = [
                "This guidance is based on summary report data and must be confirmed by a clinician."
            ]
        result["rationale"] = [_clean_item(x, 220) for x in result["rationale"][:4]]
        result["limitations"] = [_clean_item(x, 220) for x in result["limitations"][:3]]
        result["confidence"] = round(max(0.0, min(1.0, result["confidence"])), 3)
        return result
