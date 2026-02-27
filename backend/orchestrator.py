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
        case = self.get_case(case_id)
        report_text = str(case.get("report_text", ""))
        diagnosis_context = case.get("diagnosis_context", [])
        lab_context = case.get("lab_context", [])
        lab_context_text = str(case.get("lab_context_text", ""))
        previous_report_text = str(case.get("previous_report_text", "") or "")
        enriched_report_text = self._build_enriched_report_text(
            report_text=report_text,
            diagnosis_context=diagnosis_context,
            lab_context_text=lab_context_text,
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
            "diagnosis_items_count": len(diagnosis_context),
            "lab_items_count": len(lab_context),
        }
        state["report_text_enriched"] = enriched_report_text
        state["previous_report_text"] = previous_report_text
        final_state = self.flow.run(state)
        if self.settings.strict_agent_validation:
            self._validate_state(final_state)
        return final_state

    @staticmethod
    def _build_enriched_report_text(
        report_text: str,
        diagnosis_context: list[Any],
        lab_context_text: str,
    ) -> str:
        parts = [f"DISCHARGE_SUMMARY:\n{report_text}"]
        if diagnosis_context:
            parts.append(f"DIAGNOSES_ICD_CONTEXT:\n{'; '.join(str(x) for x in diagnosis_context)}")
        if lab_context_text:
            parts.append(f"LABEVENTS_CONTEXT:\n{lab_context_text}")
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
        state["clinical_summary"] = self.clinical_agent.run(state["report_text_enriched"])
        return state

    def _step_risk(self, state: dict[str, Any]) -> dict[str, Any]:
        state["risk"] = self.risk_agent.run(
            state["report_text_enriched"], state["clinical_summary"]
        )
        return state

    def _step_explainability(self, state: dict[str, Any]) -> dict[str, Any]:
        state["explanation"] = self.explainability_agent.run(
            state["report_text_enriched"], state["clinical_summary"], state["risk"]
        )
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

        state["recommendation"] = self.recommendation_agent.run(
            state["report_text_enriched"], state["clinical_summary"], state["risk"]
        )
        state["justification"] = self.justification_agent.run(
            state["recommendation"], state["explanation"], state["risk"]
        )
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
