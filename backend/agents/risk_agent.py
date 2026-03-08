from __future__ import annotations

import json
import re

from backend.model_loader import generate_reasoning
from backend.utils.logger import get_logger
from backend.utils.output_validator import parse_json_object, validate_json_format

logger = get_logger(__name__)


def _primary_diagnosis_text(report_text: str, clinical_summary: dict) -> str:
    match = re.search(r"primary diagnosis:\s*([^.]+)", report_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    diagnosis = clinical_summary.get("diagnosis", []) if isinstance(clinical_summary, dict) else []
    if diagnosis:
        return str(diagnosis[0]).strip()
    return ""


def _heuristic_risk(report_text: str, clinical_summary: dict) -> dict:
    text = report_text.lower()
    diagnosis_text = _primary_diagnosis_text(report_text, clinical_summary).lower()
    score = 0
    reasons: list[str] = []

    high_signals = {
        "sepsis": 3,
        "septicemia": 3,
        "respiratory failure": 3,
        "cardiogenic shock": 4,
        "shock": 3,
        "acute kidney failure": 2,
        "hepatic encephalopathy": 2,
        "subendocardial infarction": 3,
        "pulmonary embolism": 3,
    }
    medium_signals = {
        "hepatitis": 2,
        "alcoholic hepatitis": 2,
        "pancreatitis": 2,
        "pneumonia": 2,
        "atrial fibrillation": 1,
        "thrombocytopenia": 1,
        "chronic kidney disease": 1,
    }
    lower_signals = {
        "fracture": -1,
        "humeral fracture": -2,
        "closed fracture": -1,
    }

    for k, w in high_signals.items():
        if k in text or k in diagnosis_text:
            score += w
            reasons.append(f"High-acuity indicator present: {k}.")
    for k, w in medium_signals.items():
        if k in text or k in diagnosis_text:
            score += w
            reasons.append(f"Moderate-risk condition present: {k}.")
    for k, w in lower_signals.items():
        if k in text or k in diagnosis_text:
            score += w
            reasons.append(f"Lower acute systemic risk pattern: {k}.")

    if "acute" in text:
        score += 1
    if "chronic" in text:
        score += 1
    if "admitted to icu" in text or "critical care" in text:
        score += 2

    if score >= 5:
        level = "High"
    elif score >= 2:
        level = "Medium"
    else:
        level = "Low"

    if not reasons:
        reasons = ["Risk estimated from available report context and extracted findings."]

    return {
        "risk_level": level,
        "reasons": reasons[:4],
        "scope_note": "Risk estimate only; not a diagnosis.",
    }


class RiskAgent:
    """FLAN-T5 risk stratification agent."""

    def run(self, report_text: str, clinical_summary: dict) -> dict:
        prompt = (
            "You are a clinical risk assessment assistant.\n"
            "Task:\n"
            "1) Assign risk_level as exactly one of: Low, Medium, or High.\n"
            "2) Provide exactly 3 reasons. Each reason MUST be a single concrete clinical observation.\n"
            "   Do NOT repeat the risk level inside a reason.\n"
            "   Do NOT use vague phrases like 'as mentioned' or 'based on report'.\n"
            "3) Do not diagnose and do not prescribe.\n"
            "Return JSON ONLY with keys: risk_level (string), reasons (list of 3 strings), scope_note (string).\n"
            "Do NOT include any text before or after the JSON.\n"
            f"REPORT:\n{report_text}\n"
            f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}"
        )
        logger.debug("RiskAgent: attempt 1")
        try:
            parsed = parse_json_object(generate_reasoning(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.error("RiskAgent generate_reasoning failed: %s", exc)
            parsed = {}
        if not validate_json_format(parsed):
            logger.debug("RiskAgent: attempt 2 (json format failed)")
            try:
                parsed = parse_json_object(generate_reasoning(prompt))
            except Exception as exc:  # noqa: BLE001
                logger.error("RiskAgent generate_reasoning failed on attempt 2: %s", exc)
                parsed = {}
        risk_level = str(parsed.get("risk_level", "Medium")).strip().title()
        if risk_level not in {"Low", "Medium", "High"}:
            risk_level = "Medium"

        result = {
            "risk_level": risk_level,
            "reasons": [str(x) for x in parsed.get("reasons", []) if str(x).strip()],
            "scope_note": str(parsed.get("scope_note", "Risk estimate only; not a diagnosis.")).strip(),
        }
        result["reasons"] = result["reasons"][:3]
        if len(result["reasons"]) < 3:
            heuristic = _heuristic_risk(report_text, clinical_summary)
            result["risk_level"] = heuristic["risk_level"]
            result["reasons"] = heuristic["reasons"][:3]
            if not result["scope_note"]:
                result["scope_note"] = heuristic["scope_note"]
        return result
