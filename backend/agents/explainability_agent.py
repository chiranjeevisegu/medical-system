from __future__ import annotations

import json
import re

from backend.model_loader import generate_reasoning
from backend.utils.logger import get_logger
from backend.utils.output_validator import coverage_score, parse_json_object, validate_json_format

logger = get_logger(__name__)


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
    output = " ".join(
        [
            str(result.get("simple_explanation", "")),
            str(result.get("physiological_process", "")),
            str(result.get("plain_language_takeaway", "")),
        ]
    )
    return coverage_score(diagnoses[:3], output)


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
            "Write at a 6th-8th grade reading level. Avoid ALL medical jargon.\n"
            "You MUST cover ALL FOUR of the following points:\n"
            "1. Main disease: what the primary condition is, in simple plain language.\n"
            "2. Body impact: how the disease affects the body (physiological process), explained simply.\n"
            "3. Possible risks and complications the patient should know about.\n"
            "4. Why follow-up care and monitoring are important.\n"
            "You MUST mention ALL major diagnoses listed below.\n"
            "Return JSON ONLY with exactly these three keys:\n"
            "{\"simple_explanation\": \"...\", \"physiological_process\": \"...\", \"plain_language_takeaway\": \"\"}\n"
            "Do NOT include any text before or after the JSON.\n"
            f"MAJOR_DIAGNOSES:\n{json.dumps(diagnosis_list, ensure_ascii=False)}\n"
            f"REPORT:\n{report_text}\n"
            f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}\n"
            f"RISK:\n{json.dumps(risk, ensure_ascii=False)}"
        )
        logger.debug("ExplainabilityAgent: attempt 1")
        try:
            parsed = parse_json_object(generate_reasoning(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.error("ExplainabilityAgent: generate_reasoning failed on attempt 1: %s", exc)
            parsed = {}
        if not validate_json_format(parsed):
            logger.debug("ExplainabilityAgent: attempt 2 (json format failed)")
            try:
                parsed = parse_json_object(generate_reasoning(prompt))
            except Exception as exc:  # noqa: BLE001
                logger.error("ExplainabilityAgent: generate_reasoning failed on attempt 2: %s", exc)
                parsed = {}
        result = {
            "simple_explanation": str(parsed.get("simple_explanation", "")).strip(),
            "physiological_process": str(parsed.get("physiological_process", "")).strip(),
            "plain_language_takeaway": str(parsed.get("plain_language_takeaway", "")).strip(),
        }
        if not _is_valid_explanation(result) or _explanation_coverage_score(clinical_summary, result) < 0.5:
            logger.debug("ExplainabilityAgent: coverage-retry")
            try:
                retry = parse_json_object(generate_reasoning(prompt))
            except Exception as exc:  # noqa: BLE001
                logger.error("ExplainabilityAgent coverage-retry failed: %s", exc)
                retry = {}
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
