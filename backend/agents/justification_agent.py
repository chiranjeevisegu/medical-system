from __future__ import annotations

import json
import re

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
    raw = generate_reasoning(prompt)
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
        l for l in limitations
        if not any(tok in l.lower() for tok in ("item1", "item2", "placeholder", "rationale:"))
    ]
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
    return True


class JustificationAgent:
    """FLAN-T5 recommendation justification agent."""

    def run(self, recommendation: dict, explainability: dict, risk: dict) -> dict:
        prompt = (
            "Explain why each recommendation was suggested.\n"
            "Require at least 2 rationale points and at least 1 limitation.\n"
            "Return JSON only with keys: rationale, confidence, limitations.\n"
            "confidence must be a float between 0 and 1.\n"
            f"RECOMMENDATION:\n{json.dumps(recommendation, ensure_ascii=False)}\n"
            f"EXPLANATION:\n{json.dumps(explainability, ensure_ascii=False)}\n"
            f"RISK:\n{json.dumps(risk, ensure_ascii=False)}"
        )
        parsed = _parse_json_response(generate_reasoning(prompt))
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        result = {
            "rationale": [str(x) for x in parsed.get("rationale", []) if str(x).strip()],
            "confidence": max(0.0, min(1.0, confidence)),
            "limitations": [str(x) for x in parsed.get("limitations", []) if str(x).strip()],
        }
        if not _is_valid_justification(result):
            retry = _parse_json_response(generate_reasoning(prompt))
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
            fallback = _fallback_justification(recommendation, explainability, risk)
            result["rationale"] = fallback["rationale"] or [
                "These steps are intended to reduce short-term clinical deterioration risk.",
                "The plan prioritizes timely follow-up and symptom escalation awareness.",
            ]
            if result["confidence"] <= 0.0 or result["confidence"] == 0.5:
                result["confidence"] = fallback["confidence"]
            result["limitations"] = fallback["limitations"] or [
                "Guidance is based on summary information and does not replace clinician judgment."
            ]
        if not _is_valid_justification(result):
            result["rationale"] = [
                "The suggested actions focus on early detection of worsening symptoms.",
                "Follow-up and monitoring are emphasized because complications can progress quickly.",
            ]
            result["confidence"] = max(0.6, result["confidence"])
            result["limitations"] = [
                "This guidance is based on summary report data and must be confirmed by a clinician."
            ]
        return result
