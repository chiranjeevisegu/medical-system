from __future__ import annotations

import json
import re

from backend.model_loader import generate_reasoning
from backend.utils.logger import get_logger
from backend.utils.output_validator import parse_json_object, validate_recommendation_output
from backend.utils.text_cleaner import clean_list_items

logger = get_logger(__name__)


def _clean_item(text: str, max_chars: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -\t")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return cleaned


def _normalize_items(items: list[str], max_items: int) -> list[str]:
    out: list[str] = []
    for raw in items:
        item = _clean_item(raw)
        if not item:
            continue
        if "discharge summary for hospital admission" in item.lower():
            continue
        low = item.lower()
        if (
            "risk_level" in low
            or "the following is" in low
            or "json" in low
            or "{" in item
            or "}" in item
            or len(item) > 170
        ):
            continue
        if item not in out:
            out.append(item)
        if len(out) >= max_items:
            break
    return out


def _contains_placeholder(value: str) -> bool:
    low = value.lower()
    return any(tok in low for tok in ["item1", "item2", "item3", "placeholder", "..."])


def _is_valid_recommendation(result: dict) -> bool:
    safe = [str(x).strip() for x in result.get("safe_next_steps", []) if str(x).strip()]
    urgent = [str(x).strip() for x in result.get("urgent_attention_signs", []) if str(x).strip()]
    if len(safe) < 3 or len(urgent) < 2:
        return False
    joined = " ".join(safe + urgent + [str(result.get("boundaries", ""))])
    if _contains_placeholder(joined):
        return False
    is_ok, _ = validate_recommendation_output(result)
    return is_ok


def _fallback_recommendation(report_text: str, clinical_summary: dict, risk: dict) -> dict:
    prompt = (
        "Generate patient-safe recommendations.\n"
        "Output exactly in this line format:\n"
        "SAFE_NEXT_STEPS: item1 | item2 | item3\n"
        "URGENT_SIGNS: item1 | item2\n"
        "BOUNDARIES: ...\n"
        "No prescriptions, no dosage, no diagnosis claims.\n"
        f"REPORT:\n{report_text}\n"
        f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}\n"
        f"RISK:\n{json.dumps(risk, ensure_ascii=False)}"
    )
    raw = generate_reasoning(prompt)
    safe: list[str] = []
    urgent: list[str] = []
    boundaries = ""
    for line in raw.splitlines():
        low = line.lower().strip()
        if low.startswith("safe_next_steps:"):
            items = line.split(":", 1)[1].split("|")
            safe = [x.strip(" -\t") for x in items if x.strip()]
        elif low.startswith("urgent_signs:"):
            items = line.split(":", 1)[1].split("|")
            urgent = [x.strip(" -\t") for x in items if x.strip()]
        elif low.startswith("boundaries:"):
            boundaries = line.split(":", 1)[1].strip()

    if not safe and not urgent:
        lines = []
        for line in raw.splitlines():
            clean = line.strip(" -\t")
            if not clean:
                continue
            low = clean.lower()
            if any(low.startswith(p) for p in ("safe_next_steps:", "urgent_signs:", "boundaries:")):
                continue
            lines.append(clean)
        for item in lines:
            low = item.lower()
            if any(k in low for k in ("urgent", "emergency", "immediately", "severe", "call")):
                if item not in urgent:
                    urgent.append(item)
            else:
                if item not in safe:
                    safe.append(item)
            if len(safe) >= 4 and len(urgent) >= 2:
                break

    if len(safe) < 3:
        safe_seed = [
            "Schedule a follow-up visit with your clinician within the next few days.",
            "Track key symptoms daily and write down any worsening changes.",
            "Complete the follow-up tests recommended in your discharge plan.",
            "Review your current medications and warning signs with your care team.",
        ]
        safe = (safe + safe_seed)[:4]
    if len(urgent) < 2:
        urgent_seed = [
            "Go to urgent care for persistent high fever, breathing trouble, or confusion.",
            "Seek emergency help for sudden chest pain, fainting, or severe weakness.",
        ]
        urgent = (urgent + urgent_seed)[:3]
    if not boundaries:
        boundaries = "Non-diagnostic guidance only; clinician confirmation required."
    return {
        "safe_next_steps": _normalize_items(safe, 4),
        "urgent_attention_signs": _normalize_items(urgent, 3),
        "boundaries": _clean_item(boundaries, 180),
    }


class RecommendationAgent:
    """FLAN-T5 safety-oriented recommendation agent."""

    def run(self, report_text: str, clinical_summary: dict, risk: dict) -> dict:
        prompt = (
            "You are a patient safety assistant. Generate safe, non-clinical guidance.\n"
            "STRICT RULES – you MUST follow ALL of these:\n"
            "- Do NOT diagnose the patient.\n"
            "- Do NOT prescribe any medication.\n"
            "- Do NOT mention any dosage, drug name, or drug dose.\n"
            "- Do NOT use placeholders like 'item1', 'item2', or '...'.\n"
            "REQUIRED OUTPUT:\n"
            "- Exactly 3 to 5 concrete, safe next steps the patient should take.\n"
            "- Exactly 2 warning symptoms that require urgent medical attention.\n"
            "Return JSON ONLY with these keys: safe_next_steps (list), urgent_attention_signs (list), boundaries (string).\n"
            "Do NOT include any text before or after the JSON object.\n"
            f"REPORT:\n{report_text}\n"
            f"CLINICAL_SUMMARY:\n{json.dumps(clinical_summary, ensure_ascii=False)}\n"
            f"RISK:\n{json.dumps(risk, ensure_ascii=False)}"
        )
        logger.debug("RecommendationAgent: attempt 1")
        try:
            parsed = parse_json_object(generate_reasoning(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.error("RecommendationAgent generate_reasoning failed: %s", exc)
            parsed = {}
        result = {
            "safe_next_steps": [str(x) for x in parsed.get("safe_next_steps", []) if str(x).strip()],
            "urgent_attention_signs": [str(x) for x in parsed.get("urgent_attention_signs", []) if str(x).strip()],
            "boundaries": str(parsed.get("boundaries", "")).strip(),
        }
        if not _is_valid_recommendation(result):
            logger.debug("RecommendationAgent: attempt 2 (retry)")
            try:
                retry = parse_json_object(generate_reasoning(prompt))
            except Exception as exc:  # noqa: BLE001
                logger.error("RecommendationAgent generate_reasoning failed (attempt 2): %s", exc)
                retry = {}
            retry_result = {
                "safe_next_steps": [str(x) for x in retry.get("safe_next_steps", []) if str(x).strip()],
                "urgent_attention_signs": [str(x) for x in retry.get("urgent_attention_signs", []) if str(x).strip()],
                "boundaries": str(retry.get("boundaries", "")).strip(),
            }
            if _is_valid_recommendation(retry_result):
                result = retry_result
        if not _is_valid_recommendation(result):
            fallback = _fallback_recommendation(report_text, clinical_summary, risk)
            result["safe_next_steps"] = fallback["safe_next_steps"]
            result["urgent_attention_signs"] = fallback["urgent_attention_signs"]
            result["boundaries"] = result["boundaries"] or fallback["boundaries"]
        result["safe_next_steps"] = _normalize_items(result["safe_next_steps"], 4)
        result["urgent_attention_signs"] = _normalize_items(result["urgent_attention_signs"], 3)
        result["boundaries"] = _clean_item(result["boundaries"], 180)
        if not _is_valid_recommendation(result):
            fallback = _fallback_recommendation(report_text, clinical_summary, risk)
            result["safe_next_steps"] = fallback["safe_next_steps"]
            result["urgent_attention_signs"] = fallback["urgent_attention_signs"]
            result["boundaries"] = fallback["boundaries"]
        # Final MIMIC token strip on output lists
        result["safe_next_steps"] = clean_list_items(result["safe_next_steps"])
        result["urgent_attention_signs"] = clean_list_items(result["urgent_attention_signs"])
        return result
