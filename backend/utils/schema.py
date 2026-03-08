from __future__ import annotations

from typing import Any


FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "disease_category",
        "disease_confidence",
        "risk",
        "explanation",
        "recommendation",
        "justification",
        "verification",
    ],
    "properties": {
        "disease_category": {
            "type": "string",
            "enum": ["Infectious", "Cardiovascular", "Respiratory", "Neurological",
                     "Endocrine", "Musculoskeletal", "Other"],
        },
        "disease_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "risk": {
            "type": "object",
            "required": ["risk_level", "reasons", "scope_note"],
            "properties": {
                "risk_level": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "scope_note": {"type": "string"},
            },
        },
        "explanation": {
            "type": "object",
            "required": ["simple_explanation", "physiological_process", "plain_language_takeaway"],
            "properties": {
                "simple_explanation": {"type": "string", "minLength": 30},
                "physiological_process": {"type": "string"},
                "plain_language_takeaway": {"type": "string", "minLength": 20},
            },
        },
        "recommendation": {
            "type": "object",
            "required": ["safe_next_steps", "urgent_attention_signs", "boundaries"],
            "properties": {
                "safe_next_steps": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "urgent_attention_signs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "boundaries": {"type": "string"},
            },
        },
        "justification": {
            "type": "object",
            "required": ["rationale", "confidence", "limitations"],
            "properties": {
                "rationale": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
        },
        "verification": {
            "type": "object",
            "required": ["contradictions", "consistency_score", "safety_notes"],
            "properties": {
                "contradictions": {"type": "array", "items": {"type": "string"}},
                "consistency_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "safety_notes": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def build_final_output(
    clinical_summary: dict[str, Any],
    risk: dict[str, Any],
    explanation: dict[str, Any],
    recommendation: dict[str, Any],
    justification: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Build the fixed-schema final output dict from all agent results.

    All sub-dicts are validated against minimum content rules and given
    safe hard-coded defaults where agent output is missing or empty.
    """
    valid_categories = {
        "Infectious", "Cardiovascular", "Respiratory", "Neurological",
        "Endocrine", "Musculoskeletal", "Other",
    }
    disease_category = str(clinical_summary.get("disease_category", "Other")).strip()
    if disease_category not in valid_categories:
        disease_category = "Other"

    try:
        disease_confidence = float(clinical_summary.get("disease_confidence", 0.0))
    except (TypeError, ValueError):
        disease_confidence = 0.0
    disease_confidence = round(max(0.0, min(1.0, disease_confidence)), 4)

    # --- Risk hardening ---
    risk_out = risk if isinstance(risk, dict) else {}
    risk_level = str(risk_out.get("risk_level", "Medium")).strip().title()
    if risk_level not in {"Low", "Medium", "High"}:
        risk_level = "Medium"
    risk_reasons = [str(x).strip() for x in risk_out.get("reasons", []) if str(x).strip()]
    if not risk_reasons:
        risk_reasons = ["Risk assessed from available clinical data."]
    risk_out = {
        "risk_level": risk_level,
        "reasons": risk_reasons[:4],
        "scope_note": str(risk_out.get("scope_note", "Risk estimate only; not a diagnosis.")).strip()
                      or "Risk estimate only; not a diagnosis.",
    }

    # --- Explanation hardening ---
    expl_out = explanation if isinstance(explanation, dict) else {}
    simple = str(expl_out.get("simple_explanation", "")).strip()
    process = str(expl_out.get("physiological_process", "")).strip()
    takeaway = str(expl_out.get("plain_language_takeaway", "")).strip()
    if len(simple) < 30:
        simple = (simple + " Your clinician will clarify your specific findings and next steps.").strip()
    if not process:
        process = "The condition may affect body systems involved in your diagnosis."
    if len(takeaway) < 20:
        takeaway = (takeaway + " Please follow up with your care team.").strip()
    expl_out = {"simple_explanation": simple, "physiological_process": process,
                "plain_language_takeaway": takeaway}

    # --- Recommendation hardening ---
    rec_out = recommendation if isinstance(recommendation, dict) else {}
    safe_steps = [str(x).strip() for x in rec_out.get("safe_next_steps", []) if str(x).strip()]
    urgent = [str(x).strip() for x in rec_out.get("urgent_attention_signs", []) if str(x).strip()]
    if not safe_steps:
        safe_steps = [
            "Schedule a follow-up visit with your clinician within the next few days.",
            "Monitor your symptoms daily and note any changes.",
        ]
    if not urgent:
        urgent = ["Seek urgent care for sudden severe symptoms such as chest pain or difficulty breathing."]
    rec_out = {
        "safe_next_steps": safe_steps[:5],
        "urgent_attention_signs": urgent[:3],
        "boundaries": str(rec_out.get("boundaries", "Non-diagnostic guidance only.")).strip()
                      or "Non-diagnostic guidance only.",
    }

    # --- Justification hardening ---
    just_out = justification if isinstance(justification, dict) else {}
    rationale = [str(x).strip() for x in just_out.get("rationale", []) if str(x).strip()]
    limitations = [str(x).strip() for x in just_out.get("limitations", []) if str(x).strip()]
    if len(rationale) < 2:
        rationale = [
            "The recommendations aim to reduce risk of clinical deterioration.",
            "Follow-up and monitoring enable early detection of complications.",
        ]
    if not limitations:
        limitations = ["Guidance is based on summary information and does not replace clinician judgment."]
    try:
        justification_confidence = float(just_out.get("confidence", 0.7))
    except (TypeError, ValueError):
        justification_confidence = 0.7
    just_out = {
        "rationale": rationale[:4],
        "confidence": round(max(0.0, min(1.0, justification_confidence)), 3),
        "limitations": limitations[:3],
    }

    # --- Verification hardening ---
    verif_out = verification if isinstance(verification, dict) else {}
    contradictions = [str(x).strip() for x in verif_out.get("contradictions", []) if str(x).strip()]
    safety_notes = [str(x).strip() for x in verif_out.get("safety_notes", []) if str(x).strip()]
    try:
        consistency_score = float(verif_out.get("consistency_score", 0.7))
    except (TypeError, ValueError):
        consistency_score = 0.7
    if not safety_notes:
        safety_notes = ["All outputs are non-diagnostic and require clinician review."]
    verif_out = {
        "contradictions": contradictions[:5],
        "consistency_score": round(max(0.0, min(1.0, consistency_score)), 3),
        "safety_notes": safety_notes[:5],
    }

    return {
        "disease_category": disease_category,
        "disease_confidence": disease_confidence,
        "risk": risk_out,
        "explanation": expl_out,
        "recommendation": rec_out,
        "justification": just_out,
        "verification": verif_out,
    }
