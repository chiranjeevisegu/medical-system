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
            "1) Assign risk_level as Low, Medium, or High.\n"
            "2) Provide exactly 3 short reasons.\n"
            "3) Do not diagnose and do not prescribe.\n"
            "Return JSON only with keys: risk_level, reasons, scope_note.\n"
            f"REPORT:\n{report_text}\n"
            f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}"
        )
        parsed = _parse_json_response(generate_reasoning(prompt))
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
