from __future__ import annotations

from typing import Any

from backend.agents.clinical_agent import ClinicalAgent
from backend.agents.explainability_agent import ExplainabilityAgent
from backend.agents.justification_agent import JustificationAgent
from backend.agents.recommendation_agent import RecommendationAgent
from backend.agents.risk_agent import RiskAgent
from backend.agents.verification_agent import VerificationAgent
from backend.config import Settings
from backend.dataset_loader import MIMICDatasetLoader
from backend.langchain_orchestration import SequentialAgentFlow
from backend.model_loader import ensure_clinical_classifier
from backend.utils.logger import get_logger
from backend.utils.output_validator import contains_placeholder, validate_final_output
from backend.utils.schema import build_final_output
from backend.utils.text_cleaner import clean_report_text

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = MIMICDatasetLoader(
            dataset_dir=settings.dataset_dir,
            output_path=settings.samples_path,
            max_cases=settings.max_cases,
        )
        self.samples = self.loader.prepare_samples(force_refresh=False)
        self.case_map = {sample["case_id"]: sample for sample in self.samples}
        self.classifier_status = ensure_clinical_classifier(self.samples)

        self.clinical_agent = ClinicalAgent()
        self.risk_agent = RiskAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.recommendation_agent = RecommendationAgent()
        self.justification_agent = JustificationAgent()
        self.verification_agent = VerificationAgent()
        self.flow = SequentialAgentFlow(
            [
                self._step_clinical,
                self._step_risk,
                self._step_explainability,
                self._step_pre_verification,
                self._step_recommendation_and_justification,
                self._step_final_verification,
                self._step_build_final_output,
                self._step_user_visible,
            ]
        )

    def list_case_ids(self) -> list[str]:
        return list(self.case_map.keys())

    def get_case(self, case_id: str) -> dict[str, Any]:
        case = self.case_map.get(case_id)
        if case is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        return case

    def analyze_case(self, case_id: str) -> dict[str, Any]:
        logger.info("Pipeline start – case_id=%s", case_id)
        case = self.get_case(case_id)
        report_text = str(case.get("report_text", ""))
        diagnosis_context = case.get("diagnosis_context", [])
        lab_context = case.get("lab_context", [])
        lab_context_text = str(case.get("lab_context_text", ""))
        medications_context_text = str(case.get("medications_context_text", "") or "")
        previous_report_text = str(case.get("previous_report_text", "") or "")
        enriched_report_text = self._build_enriched_report_text(
            report_text=report_text,
            diagnosis_context=diagnosis_context,
            lab_context_text=lab_context_text,
            medications_context_text=medications_context_text,
        )

        state: dict[str, Any] = {"case_id": case_id}
        state["dataset_context"] = {
            "diagnosis_context": diagnosis_context,
            "lab_context": lab_context,
        }
        state["data_sources_used"] = {
            "report_source": str(case.get("source", "")),
            "noteevents_report_used": bool(report_text.strip()),
            "diagnoses_icd_used": bool(diagnosis_context),
            "labevents_used": bool(lab_context),
            "medications_used": bool(medications_context_text.strip()),
            "diagnosis_items_count": len(diagnosis_context),
            "lab_items_count": len(lab_context),
        }
        state["report_text_enriched"] = enriched_report_text
        state["previous_report_text"] = previous_report_text
        final_state = self.flow.run(state)
        if self.settings.strict_agent_validation:
            self._validate_state(final_state)
        logger.info("Pipeline complete – case_id=%s", case_id)
        return final_state

    def analyze_report_text(self, report_text: str) -> dict[str, Any]:
        """Run the full 6-agent pipeline on freeform report text (no MIMIC case required).

        This is the entry point used by the React frontend via POST /analyze_report.
        """
        import uuid  # noqa: PLC0415
        case_id = f"web_{uuid.uuid4().hex[:8]}"
        logger.info("analyze_report_text: case_id=%s chars=%d", case_id, len(report_text))

        # Strip MIMIC anonymization tokens before any agent sees the text
        cleaned_text = clean_report_text(report_text)
        logger.debug("analyze_report_text: cleaned to %d chars", len(cleaned_text))

        enriched = self._build_enriched_report_text(
            report_text=cleaned_text,
            diagnosis_context=[],
            lab_context_text="",
            medications_context_text="",
        )

        state: dict[str, Any] = {"case_id": case_id}
        state["dataset_context"] = {"diagnosis_context": [], "lab_context": []}
        state["data_sources_used"] = {
            "report_source": "web_input",
            "noteevents_report_used": True,
            "diagnoses_icd_used": False,
            "labevents_used": False,
            "medications_used": False,
            "diagnosis_items_count": 0,
            "lab_items_count": 0,
        }
        state["report_text_enriched"] = enriched
        state["previous_report_text"] = ""

        final_state = self.flow.run(state)

        # Build and return the structured final output
        from backend.utils.schema import build_final_output  # noqa: PLC0415
        return build_final_output(
            clinical_summary=final_state.get("clinical_summary", {}),
            risk=final_state.get("risk", {}),
            explanation=final_state.get("explanation", {}),
            recommendation=final_state.get("recommendation", {}),
            justification=final_state.get("justification", {}),
            verification=final_state.get("verification", {}),
        )


    @staticmethod
    def _build_enriched_report_text(
        report_text: str,
        diagnosis_context: list[Any],
        lab_context_text: str,
        medications_context_text: str = "",
    ) -> str:
        parts = [f"DISCHARGE_SUMMARY:\n{report_text}"]
        if diagnosis_context:
            parts.append(f"DIAGNOSES_ICD_CONTEXT:\n{'; '.join(str(x) for x in diagnosis_context)}")
        if lab_context_text:
            parts.append(f"LABEVENTS_CONTEXT:\n{lab_context_text}")
        if medications_context_text:
            parts.append(f"MEDICATIONS_CONTEXT:\n{medications_context_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_user_visible_output(state: dict[str, Any]) -> dict[str, Any]:
        risk = state.get("risk", {})
        explanation = state.get("explanation", {})
        justification = state.get("justification", {})
        verification = state.get("verification", {})

        return {
            "explainability": {
                "risk_level": risk.get("risk_level", "Medium"),
                "simple_explanation": explanation.get("simple_explanation", ""),
                "physiological_process": explanation.get("physiological_process", ""),
                "plain_language_takeaway": explanation.get("plain_language_takeaway", ""),
                "disease_category": state.get("clinical_summary", {}).get("disease_category", ""),
                "disease_confidence": state.get("clinical_summary", {}).get("disease_confidence", 0.0),
            },
            "recommendation_justification": {
                "rationale": justification.get("rationale", []),
                "confidence": justification.get("confidence", 0.0),
                "limitations": justification.get("limitations", []),
            },
            "verification_summary": {
                "consistency_score": verification.get("consistency_score", 0.0),
                "contradictions": verification.get("contradictions", []),
                "safety_notes": verification.get("safety_notes", []),
            },
        }

    def _step_clinical(self, state: dict[str, Any]) -> dict[str, Any]:
        logger.info("Step clinical_agent – case_id=%s", state.get("case_id", "?"))
        state["clinical_summary"] = self.clinical_agent.run(state["report_text_enriched"])
        logger.info("clinical_agent done – category=%s conf=%.2f",
            state["clinical_summary"].get("disease_category"),
            state["clinical_summary"].get("disease_confidence", 0.0))
        return state

    def _step_risk(self, state: dict[str, Any]) -> dict[str, Any]:
        logger.info("Step risk_agent – case_id=%s", state.get("case_id", "?"))
        state["risk"] = self.risk_agent.run(
            state["report_text_enriched"], state["clinical_summary"]
        )
        logger.info("risk_agent done – risk_level=%s", state["risk"].get("risk_level"))
        return state

    def _step_explainability(self, state: dict[str, Any]) -> dict[str, Any]:
        logger.info("Step explainability_agent – case_id=%s", state.get("case_id", "?"))
        state["explanation"] = self.explainability_agent.run(
            state["report_text_enriched"], state["clinical_summary"], state["risk"]
        )
        logger.info("explainability_agent done")
        return state

    def _step_pre_verification(self, state: dict[str, Any]) -> dict[str, Any]:
        state["verification_preliminary"] = self.verification_agent.run(
            state["clinical_summary"],
            state["risk"],
            state["explanation"],
            {},
            {},
        )
        return state

    def _step_recommendation_and_justification(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        confidence_gate = float(
            state["verification_preliminary"].get("consistency_score", 0.0)
        )
        state["recommendation_mode"] = "low_confidence" if confidence_gate < 0.55 else "normal"
        logger.info("Step recommendation+justification – mode=%s case_id=%s",
            state["recommendation_mode"], state.get("case_id", "?"))

        state["recommendation"] = self.recommendation_agent.run(
            state["report_text_enriched"], state["clinical_summary"], state["risk"]
        )
        state["justification"] = self.justification_agent.run(
            state["recommendation"], state["explanation"], state["risk"]
        )
        logger.info("recommendation+justification done")
        return state

    def _step_final_verification(self, state: dict[str, Any]) -> dict[str, Any]:
        state["verification"] = self.verification_agent.run(
            state["clinical_summary"],
            state["risk"],
            state["explanation"],
            state["recommendation"],
            state["justification"],
        )
        return state

    def _step_user_visible(self, state: dict[str, Any]) -> dict[str, Any]:
        state["user_visible"] = self._build_user_visible_output(state)
        return state

    def _step_build_final_output(self, state: dict[str, Any]) -> dict[str, Any]:
        state["final_output"] = build_final_output(
            clinical_summary=state.get("clinical_summary", {}),
            risk=state.get("risk", {}),
            explanation=state.get("explanation", {}),
            recommendation=state.get("recommendation", {}),
            justification=state.get("justification", {}),
            verification=state.get("verification", {}),
        )
        ok, issues = validate_final_output(state["final_output"])
        if not ok:
            logger.warning("Validation failed (attempt 1) – issues=%s – case_id=%s regenerating…",
                issues, state.get("case_id", "?"))
            # One explicit regeneration pass at orchestration level.
            state["recommendation"] = self.recommendation_agent.run(
                state["report_text_enriched"], state["clinical_summary"], state["risk"]
            )
            state["justification"] = self.justification_agent.run(
                state["recommendation"], state["explanation"], state["risk"]
            )
            state["verification"] = self.verification_agent.run(
                state["clinical_summary"],
                state["risk"],
                state["explanation"],
                state["recommendation"],
                state["justification"],
            )
            state["final_output"] = build_final_output(
                clinical_summary=state.get("clinical_summary", {}),
                risk=state.get("risk", {}),
                explanation=state.get("explanation", {}),
                recommendation=state.get("recommendation", {}),
                justification=state.get("justification", {}),
                verification=state.get("verification", {}),
            )

        ok, issues2 = validate_final_output(state["final_output"])
        if not ok:
            logger.warning("Validation failed (attempt 2) – issues=%s – case_id=%s applying safety net",
                issues2, state.get("case_id", "?"))
            # Deterministic final safety net to avoid missing/placeholder fields.
            recommendation = state["final_output"].get("recommendation", {})
            safe_next = recommendation.get("safe_next_steps", [])
            if not isinstance(safe_next, list) or not safe_next or contains_placeholder(safe_next):
                recommendation["safe_next_steps"] = [
                    "Schedule follow-up with your clinician within a few days.",
                    "Track symptoms daily and report worsening changes.",
                    "Complete the follow-up tests listed in your discharge plan.",
                ]
            urgent = recommendation.get("urgent_attention_signs", [])
            if not isinstance(urgent, list) or len(urgent) < 2 or contains_placeholder(urgent):
                recommendation["urgent_attention_signs"] = [
                    "Seek urgent care for breathing difficulty, confusion, or persistent fever.",
                    "Seek emergency help for chest pain, fainting, or sudden severe weakness.",
                ]
            recommendation["boundaries"] = "Non-diagnostic guidance only; clinician confirmation required."
            state["final_output"]["recommendation"] = recommendation

            justification = state["final_output"].get("justification", {})
            rationale = justification.get("rationale", [])
            if not isinstance(rationale, list) or len([x for x in rationale if str(x).strip()]) < 2 or contains_placeholder(rationale):
                justification["rationale"] = [
                    "Recommendations prioritize early detection of worsening symptoms.",
                    "Close follow-up is advised because complications can progress quickly.",
                ]
            limitations = justification.get("limitations", [])
            if not isinstance(limitations, list) or not limitations:
                limitations = ["Guidance is based on report summary and must be confirmed clinically."]
            justification["limitations"] = limitations
            try:
                confidence = float(justification.get("confidence", 0.6))
            except Exception:
                confidence = 0.6
            justification["confidence"] = max(0.0, min(1.0, confidence))
            state["final_output"]["justification"] = justification
        return state

    @staticmethod
    def _validate_state(state: dict[str, Any]) -> None:
        issues: list[str] = []

        clinical = state.get("clinical_summary", {})
        clinical_items = sum(
            len(clinical.get(key, []))
            for key in ("diagnosis", "abnormal_findings", "key_observations")
            if isinstance(clinical.get(key, []), list)
        )
        if clinical_items == 0:
            issues.append("clinical_summary has no extracted items")
        if not str(clinical.get("disease_category", "")).strip():
            issues.append("clinical_summary.disease_category missing")
        try:
            conf = float(clinical.get("disease_confidence", 0.0))
            if conf < 0.0 or conf > 1.0:
                issues.append("clinical_summary.disease_confidence out of range")
        except Exception:
            issues.append("clinical_summary.disease_confidence invalid")

        risk = state.get("risk", {})
        if str(risk.get("risk_level", "")).strip() not in {"Low", "Medium", "High"}:
            issues.append("risk_level missing or invalid")
        if not str(risk.get("scope_note", "")).strip():
            issues.append("risk.scope_note missing")

        explanation = state.get("explanation", {})
        if not str(explanation.get("simple_explanation", "")).strip():
            issues.append("explanation.simple_explanation missing")
        if not str(explanation.get("plain_language_takeaway", "")).strip():
            issues.append("explanation.plain_language_takeaway missing")

        verification = state.get("verification", {})
        score = verification.get("consistency_score", None)
        try:
            score_float = float(score)
            if not (0.0 <= score_float <= 1.0):
                issues.append("verification.consistency_score out of range")
        except Exception:
            issues.append("verification.consistency_score missing or invalid")

        recommendation = state.get("recommendation", {})
        boundaries = str(recommendation.get("boundaries", ""))
        halted = "halted due to low confidence" in boundaries.lower()
        if not halted:
            next_steps = recommendation.get("safe_next_steps", [])
            urgent = recommendation.get("urgent_attention_signs", [])
            if not isinstance(next_steps, list) or not isinstance(urgent, list):
                issues.append("recommendation lists malformed")
            elif len(next_steps) + len(urgent) == 0:
                issues.append("recommendation has no actionable items")

        if issues:
            raise RuntimeError("Strict agent validation failed: " + "; ".join(issues))

        final_output = state.get("final_output", {})
        ok_final, final_issues = validate_final_output(final_output if isinstance(final_output, dict) else {})
        if not ok_final:
            raise RuntimeError("Final output schema validation failed: " + "; ".join(final_issues))
