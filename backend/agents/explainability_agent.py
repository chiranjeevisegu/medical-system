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


def _clean_sentence(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return cleaned


def _fallback_explanation(report_text: str, clinical_summary: dict, risk: dict) -> dict:
    prompt = (
        "Create a patient-friendly explanation from this medical report context.\n"
        "Output exactly three labeled lines:\n"
        "SIMPLE: ...\n"
        "PROCESS: ...\n"
        "TAKEAWAY: ...\n"
        "Do not output JSON.\n"
        f"REPORT:\n{report_text}\n"
        f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}\n"
        f"RISK:\n{json.dumps(risk, ensure_ascii=False)}"
    )
    raw = generate_reasoning(prompt)
    simple = ""
    process = ""
    takeaway = ""
    for line in raw.splitlines():
        low = line.lower().strip()
        if low.startswith("simple:"):
            simple = line.split(":", 1)[1].strip()
        elif low.startswith("process:"):
            process = line.split(":", 1)[1].strip()
        elif low.startswith("takeaway:"):
            takeaway = line.split(":", 1)[1].strip()
    if not simple and raw.strip():
        simple = raw.strip().splitlines()[0].strip()
    if not process:
        lines = [ln.strip(" -\t") for ln in raw.splitlines() if ln.strip()]
        if len(lines) > 1:
            process = lines[1]
    if not takeaway:
        tail_prompt = (
            "Write one plain-language patient takeaway sentence from this explanation.\n"
            f"TEXT:\n{raw}\n"
        )
        tail = generate_reasoning(tail_prompt).strip()
        takeaway = tail or (simple if simple else "Follow up with your clinician for review.")
    return {
        "simple_explanation": _clean_sentence(simple, 320),
        "physiological_process": _clean_sentence(process, 260),
        "plain_language_takeaway": _clean_sentence(takeaway, 220),
    }


def _is_valid_explanation(result: dict) -> bool:
    if not str(result.get("simple_explanation", "")).strip():
        return False
    if not str(result.get("plain_language_takeaway", "")).strip():
        return False
    text = " ".join(
        [
            str(result.get("simple_explanation", "")),
            str(result.get("physiological_process", "")),
            str(result.get("plain_language_takeaway", "")),
        ]
    ).lower()
    bad_tokens = ["item1", "item2", "lorem", "placeholder"]
    if any(tok in text for tok in bad_tokens):
        return False
    if "associated icd diagnoses" in text:
        return False
    return True


def _explanation_coverage_score(clinical_summary: dict, result: dict) -> float:
    diagnoses = [str(x).strip() for x in clinical_summary.get("diagnosis", []) if str(x).strip()]
    if not diagnoses:
        return 1.0
    max_check = min(3, len(diagnoses))
    to_check = diagnoses[:max_check]
    output = " ".join(
        [
            str(result.get("simple_explanation", "")),
            str(result.get("physiological_process", "")),
            str(result.get("plain_language_takeaway", "")),
        ]
    ).lower()
    hits = 0
    for d in to_check:
        d_tokens = [tok for tok in re.findall(r"[a-zA-Z0-9]+", d.lower()) if len(tok) > 3]
        if not d_tokens:
            continue
        overlap = sum(1 for tok in d_tokens if tok in output)
        if overlap / len(d_tokens) >= 0.4:
            hits += 1
    return hits / max(1, len(to_check))


def _heuristic_explanation(clinical_summary: dict, risk: dict) -> dict:
    diagnosis_items = [str(x) for x in clinical_summary.get("diagnosis", [])[:3] if str(x).strip()]
    diagnosis = ", ".join(diagnosis_items[:2])
    dlow = diagnosis.lower()
    if "sepsis" in dlow:
        associated = diagnosis_items[1:3]
        assoc_text = (
            f" Associated conditions include {', '.join(associated)}."
            if associated
            else ""
        )
        simple = (
            "Sepsis means the body is having a severe whole-body response to an infection. "
            f"This can become dangerous quickly and needs close follow-up.{assoc_text}"
        )
        process = (
            "Infection can trigger widespread inflammation, which may reduce blood flow to organs such as "
            "the kidneys, heart, and brain."
        )
        takeaway = "This is a serious infection-related condition, so monitor symptoms closely and follow up quickly."
    else:
        risk_level = str(risk.get("risk_level", "Medium"))
        assoc_text = f" Main reported conditions include {diagnosis}." if diagnosis else ""
        simple = (
            "Your discharge report shows important medical findings that need close follow-up with your clinician."
            f"{assoc_text}"
        )
        process = (
            "These conditions can stress body systems over time, so monitoring and timely review help prevent worsening."
        )
        takeaway = f"Your current risk is {risk_level}; keep follow-up and act early if symptoms worsen."
    return {
        "simple_explanation": simple,
        "physiological_process": process,
        "plain_language_takeaway": takeaway,
    }


class ExplainabilityAgent:
    """FLAN-T5 plain-language explanation agent."""

    def run(self, report_text: str, clinical_summary: dict, risk: dict) -> dict:
        diagnosis_list = [str(x) for x in clinical_summary.get("diagnosis", []) if str(x).strip()][:5]
        prompt = (
            "You are explaining a hospital discharge report to a patient.\n"
            "You must mention all major diagnoses provided below.\n"
            "Explain:\n"
            "1) Primary diagnosis.\n"
            "2) At least two associated conditions.\n"
            "3) One complication risk.\n"
            "4) How it affects the body.\n"
            "Write in simple language (grade 6-8), avoid jargon.\n"
            "Return JSON only with keys: simple_explanation, physiological_process, plain_language_takeaway.\n"
            f"MAJOR_DIAGNOSES:\n{json.dumps(diagnosis_list, ensure_ascii=False)}\n"
            f"REPORT:\n{report_text}\n"
            f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}\n"
            f"RISK:\n{json.dumps(risk, ensure_ascii=False)}"
        )
        parsed = _parse_json_response(generate_reasoning(prompt))
        result = {
            "simple_explanation": str(parsed.get("simple_explanation", "")).strip(),
            "physiological_process": str(parsed.get("physiological_process", "")).strip(),
            "plain_language_takeaway": str(parsed.get("plain_language_takeaway", "")).strip(),
        }
        if not _is_valid_explanation(result) or _explanation_coverage_score(clinical_summary, result) < 0.5:
            retry = _parse_json_response(generate_reasoning(prompt))
            retry_result = {
                "simple_explanation": str(retry.get("simple_explanation", "")).strip(),
                "physiological_process": str(retry.get("physiological_process", "")).strip(),
                "plain_language_takeaway": str(retry.get("plain_language_takeaway", "")).strip(),
            }
            if _is_valid_explanation(retry_result) and _explanation_coverage_score(clinical_summary, retry_result) >= 0.5:
                result = retry_result
        if not _is_valid_explanation(result) or _explanation_coverage_score(clinical_summary, result) < 0.5:
            fallback = _fallback_explanation(report_text, clinical_summary, risk)
            result["simple_explanation"] = result["simple_explanation"] or fallback["simple_explanation"]
            result["physiological_process"] = (
                result["physiological_process"] or fallback["physiological_process"]
            )
            result["plain_language_takeaway"] = (
                result["plain_language_takeaway"] or fallback["plain_language_takeaway"]
            )
        if not _is_valid_explanation(result) or _explanation_coverage_score(clinical_summary, result) < 0.5:
            result = _heuristic_explanation(clinical_summary, risk)
        result["simple_explanation"] = _clean_sentence(result["simple_explanation"], 320)
        result["physiological_process"] = _clean_sentence(result["physiological_process"], 260)
        result["plain_language_takeaway"] = _clean_sentence(result["plain_language_takeaway"], 220)
        return result
