from __future__ import annotations

from backend.model_loader import extract_clinical_facts, predict_disease_category
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_CATEGORIES = {
    "Infectious", "Cardiovascular", "Respiratory", "Neurological",
    "Endocrine", "Musculoskeletal", "Other",
}


class ClinicalAgent:
    """BioClinicalBERT-based clinical extraction and classification agent."""

    def run(self, report_text: str) -> dict:
        logger.debug("ClinicalAgent: extracting clinical facts")
        try:
            result = extract_clinical_facts(report_text)
        except Exception as exc:  # noqa: BLE001
            logger.error("ClinicalAgent extract_clinical_facts failed: %s", exc)
            result = {}

        logger.debug("ClinicalAgent: predicting disease category")
        try:
            category = predict_disease_category(report_text)
        except Exception as exc:  # noqa: BLE001
            logger.error("ClinicalAgent predict_disease_category failed: %s", exc)
            category = {"disease_category": "Other", "disease_confidence": 0.0}

        disease_category = str(category.get("disease_category", "Other")).strip()
        if disease_category not in _VALID_CATEGORIES:
            disease_category = "Other"

        try:
            disease_confidence = float(category.get("disease_confidence", 0.0))
        except (TypeError, ValueError):
            disease_confidence = 0.0
        disease_confidence = round(max(0.0, min(1.0, disease_confidence)), 4)

        diagnosis = [str(x).strip() for x in result.get("diagnosis", []) if str(x).strip()]
        abnormal_findings = [str(x).strip() for x in result.get("abnormal_findings", []) if str(x).strip()]
        key_observations = [str(x).strip() for x in result.get("key_observations", []) if str(x).strip()]

        logger.info("ClinicalAgent done – category=%s conf=%.4f diagnoses=%d",
                    disease_category, disease_confidence, len(diagnosis))
        return {
            "diagnosis": diagnosis[:8],
            "abnormal_findings": abnormal_findings[:6],
            "key_observations": key_observations[:6],
            "disease_category": disease_category,
            "disease_confidence": disease_confidence,
        }
