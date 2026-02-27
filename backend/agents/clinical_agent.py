from __future__ import annotations

from backend.model_loader import extract_clinical_facts, predict_disease_category


class ClinicalAgent:
    """BioClinicalBERT-based clinical extraction agent."""

    def run(self, report_text: str) -> dict:
        result = extract_clinical_facts(report_text)
        category = predict_disease_category(report_text)
        return {
            "diagnosis": [str(x) for x in result.get("diagnosis", []) if str(x).strip()],
            "abnormal_findings": [str(x) for x in result.get("abnormal_findings", []) if str(x).strip()],
            "key_observations": [str(x) for x in result.get("key_observations", []) if str(x).strip()],
            "disease_category": str(category.get("disease_category", "Infectious")),
            "disease_confidence": float(category.get("disease_confidence", 0.0)),
        }
