from __future__ import annotations

import json
import re
from typing import Any

from backend.utils.schema import FINAL_OUTPUT_SCHEMA


_PLACEHOLDER_TOKENS = ("item1", "item2", "item3", "placeholder")


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(contains_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(v) for v in value)
    text = str(value).lower()
    return any(tok in text for tok in _PLACEHOLDER_TOKENS)


def coverage_score(diagnoses: list[str], explanation_text: str) -> float:
    cleaned = [str(x).strip() for x in diagnoses if str(x).strip()]
    if not cleaned:
        return 1.0
    normalized = str(explanation_text or "").lower()
    hits = 0
    for diag in cleaned:
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", diag.lower()) if len(t) > 3]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in normalized)
        if overlap / len(tokens) >= 0.4:
            hits += 1
    return hits / max(1, len(cleaned))


def validate_json_format(payload: Any) -> bool:
    return isinstance(payload, dict)


def validate_recommendation_output(recommendation: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not validate_json_format(recommendation):
        return False, ["recommendation is not valid JSON object"]
    safe_next_steps = recommendation.get("safe_next_steps", [])
    if not isinstance(safe_next_steps, list) or len([x for x in safe_next_steps if str(x).strip()]) == 0:
        issues.append("safe_next_steps not empty check failed")
    if contains_placeholder(recommendation):
        issues.append("placeholder text detected")
    return len(issues) == 0, issues


def validate_justification_output(justification: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not validate_json_format(justification):
        return False, ["justification is not valid JSON object"]
    rationale = justification.get("rationale", [])
    if not isinstance(rationale, list) or len([x for x in rationale if str(x).strip()]) < 2:
        issues.append("justification rationale length < 2")
    if contains_placeholder(justification):
        issues.append("placeholder text detected")
    return len(issues) == 0, issues


def validate_final_output(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not validate_json_format(payload):
        return False, ["final output is not valid JSON object"]

    required = FINAL_OUTPUT_SCHEMA.get("required", [])
    for key in required:
        if key not in payload:
            issues.append(f"missing required key: {key}")

    # --- field-level hardening checks ---
    # disease_confidence must be a float in [0, 1]
    dc = payload.get("disease_confidence")
    try:
        dc_float = float(dc)
        if not (0.0 <= dc_float <= 1.0):
            issues.append("disease_confidence out of range [0,1]")
    except (TypeError, ValueError):
        issues.append("disease_confidence is not a valid float")

    # risk_level must be Low / Medium / High
    risk_obj = payload.get("risk", {})
    if isinstance(risk_obj, dict):
        rl = str(risk_obj.get("risk_level", "")).strip().title()
        if rl not in {"Low", "Medium", "High"}:
            issues.append(f"risk.risk_level invalid: '{rl}'")

    # explanation text length checks
    expl = payload.get("explanation", {})
    if isinstance(expl, dict):
        simple = str(expl.get("simple_explanation", "")).strip()
        takeaway = str(expl.get("plain_language_takeaway", "")).strip()
        if len(simple) < 30:
            issues.append("explanation.simple_explanation too short (<30 chars)")
        if len(takeaway) < 20:
            issues.append("explanation.plain_language_takeaway too short (<20 chars)")

    recommendation = payload.get("recommendation", {})
    ok_rec, rec_issues = validate_recommendation_output(recommendation if isinstance(recommendation, dict) else {})
    if not ok_rec:
        issues.extend(rec_issues)

    justification = payload.get("justification", {})
    ok_jus, jus_issues = validate_justification_output(justification if isinstance(justification, dict) else {})
    if not ok_jus:
        issues.extend(jus_issues)

    return len(issues) == 0, issues
