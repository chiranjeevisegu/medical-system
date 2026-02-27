from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import textstat

from backend.baseline import BaselineSummarizer
from backend.model_loader import CATEGORY_LABELS, generate_reasoning
from backend.orchestrator import Orchestrator


LABELS_INV = {name: idx for idx, name in enumerate(CATEGORY_LABELS)}


@dataclass
class Evaluator:
    orchestrator: Orchestrator
    baseline: BaselineSummarizer
    results_path: Path

    def evaluate_all(self, case_ids: list[str] | None = None) -> dict:
        rows: list[dict] = []
        cls_y_true: list[str] = []
        cls_y_pred: list[str] = []
        ids = case_ids if case_ids is not None else self.orchestrator.list_case_ids()[:20]
        for case_id in ids:
            case = self.orchestrator.get_case(case_id)
            report_text = str(case.get("report_text", ""))

            baseline_summary = self.baseline.summarize(report_text).get("summary", "")
            analysis = self.orchestrator.analyze_case(case_id)
            agent_explanation = self._flatten_explanation(analysis.get("explanation", {}))
            recommendation_text = self._flatten_explanation(analysis.get("recommendation", {}))
            user_visible = self._flatten_explanation(analysis.get("user_visible", {}))
            rationale_text = self._flatten_explanation(analysis.get("justification", {}))
            full_patient_output = " ".join(
                part for part in [agent_explanation, recommendation_text, rationale_text, user_visible] if part
            )

            readability_fkgl = self._safe_fkgl(full_patient_output)
            patient_comprehension = self._patient_comprehension_score(full_patient_output)
            information_coverage = self._information_coverage(analysis, full_patient_output)
            trust_calibration = self._trust_calibration_score(analysis)
            agent_disagreement_rate = self._agent_disagreement_rate(analysis)
            unsafe_recommendation_rate = self._unsafe_recommendation_rate(analysis)
            consistency_rate = self._consistency_rate(analysis)
            true_label = self._category_label_from_case(case)
            pred_label = str(analysis.get("clinical_summary", {}).get("disease_category", "")).strip()
            if true_label and pred_label:
                cls_y_true.append(true_label)
                cls_y_pred.append(pred_label)

            rows.append(
                {
                    "case_id": case_id,
                    "original_fkgl": self._safe_fkgl(report_text),
                    "baseline_fkgl": self._safe_fkgl(baseline_summary),
                    "agent_fkgl": self._safe_fkgl(agent_explanation),
                    "readability_fkgl": readability_fkgl,
                    "original_length": self._word_count(report_text),
                    "baseline_length": self._word_count(baseline_summary),
                    "agent_length": self._word_count(agent_explanation),
                    "qa_score": self._qa_score(report_text, agent_explanation),
                    "patient_comprehension": patient_comprehension,
                    "information_coverage": information_coverage,
                    "trust_calibration": trust_calibration,
                    "agent_disagreement_rate": agent_disagreement_rate,
                    "unsafe_recommendation_rate": unsafe_recommendation_rate,
                    "consistency_rate": consistency_rate,
                    "classifier_true_label": true_label,
                    "classifier_pred_label": pred_label,
                    "classifier_true_id": LABELS_INV.get(true_label, -1),
                    "classifier_pred_id": LABELS_INV.get(pred_label, -1),
                }
            )

        df = pd.DataFrame(rows)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.results_path, index=False)

        if df.empty:
            return {
                "num_cases": 0,
                "avg_fkgl_improvement": 0.0,
                "avg_length_reduction": 0.0,
                "avg_qa_score": 0.0,
                "avg_patient_comprehension": 0.0,
                "avg_readability_fkgl": 0.0,
                "avg_information_coverage": 0.0,
                "avg_trust_calibration": 0.0,
                "avg_agent_disagreement_rate": 0.0,
                "avg_unsafe_recommendation_rate": 0.0,
                "avg_consistency_rate": 0.0,
                "classifier_precision": 0.0,
                "classifier_recall": 0.0,
                "classifier_f1": 0.0,
                "classifier_confusion_matrix": {},
                "std_fkgl_improvement": 0.0,
                "std_qa_score": 0.0,
                "std_information_coverage": 0.0,
                "results_path": str(self.results_path),
            }

        avg_fkgl_improvement = float((df["original_fkgl"] - df["agent_fkgl"]).mean())
        avg_length_reduction = float(
            ((df["original_length"] - df["agent_length"]) / df["original_length"].replace(0, 1)).mean()
        )
        avg_qa_score = float(df["qa_score"].mean())
        avg_patient_comprehension = float(df["patient_comprehension"].mean())
        avg_readability_fkgl = float(df["readability_fkgl"].mean())
        avg_information_coverage = float(df["information_coverage"].mean())
        avg_trust_calibration = float(df["trust_calibration"].mean())
        avg_agent_disagreement_rate = float(df["agent_disagreement_rate"].mean())
        avg_unsafe_recommendation_rate = float(df["unsafe_recommendation_rate"].mean())
        avg_consistency_rate = float(df["consistency_rate"].mean())
        cls_precision, cls_recall, cls_f1, confusion = self._classification_metrics(cls_y_true, cls_y_pred)

        return {
            "num_cases": int(len(df)),
            "avg_fkgl_improvement": avg_fkgl_improvement,
            "avg_length_reduction": avg_length_reduction,
            "avg_qa_score": avg_qa_score,
            "avg_patient_comprehension": avg_patient_comprehension,
            "avg_readability_fkgl": avg_readability_fkgl,
            "avg_information_coverage": avg_information_coverage,
            "avg_trust_calibration": avg_trust_calibration,
            "avg_agent_disagreement_rate": avg_agent_disagreement_rate,
            "avg_unsafe_recommendation_rate": avg_unsafe_recommendation_rate,
            "avg_consistency_rate": avg_consistency_rate,
            "classifier_precision": cls_precision,
            "classifier_recall": cls_recall,
            "classifier_f1": cls_f1,
            "classifier_confusion_matrix": confusion,
            "std_fkgl_improvement": float((df["original_fkgl"] - df["agent_fkgl"]).std(ddof=0)),
            "std_qa_score": float(df["qa_score"].std(ddof=0)),
            "std_information_coverage": float(df["information_coverage"].std(ddof=0)),
            "results_path": str(self.results_path),
        }

    def _patient_comprehension_score(self, text: str) -> float:
        fkgl = self._safe_fkgl(text)
        readability_component = max(0.0, min(1.0, (18.0 - fkgl) / 18.0))
        jargon_ratio = self._medical_jargon_ratio(text)
        jargon_component = max(0.0, 1.0 - jargon_ratio)
        return 0.7 * readability_component + 0.3 * jargon_component

    def _information_coverage(self, analysis: dict[str, Any], patient_text: str) -> float:
        clinical = analysis.get("clinical_summary", {})
        diagnosis_values = clinical.get("diagnosis", []) if isinstance(clinical, dict) else []
        facts: list[str] = [str(v) for v in diagnosis_values if str(v).strip()]
        if len(facts) < 3:
            for key in ("abnormal_findings", "key_observations"):
                values = clinical.get(key, [])
                if isinstance(values, list):
                    facts.extend(str(v) for v in values if str(v).strip())

        if not facts:
            return 0.0

        normalized_output = self._normalize(patient_text)
        hits = 0
        for fact in facts:
            if self._answer_in_text(self._normalize(fact), normalized_output):
                hits += 1
        return hits / len(facts)

    def _trust_calibration_score(self, analysis: dict[str, Any]) -> float:
        verification = analysis.get("verification", {})
        justification = analysis.get("justification", {})
        try:
            consistency = float(verification.get("consistency_score", 0.0))
        except (TypeError, ValueError):
            consistency = 0.0
        try:
            confidence = float(justification.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        calibration_gap = abs(confidence - consistency)
        contradiction_penalty = min(1.0, len(verification.get("contradictions", []) or []) / 5.0)
        return max(0.0, 1.0 - (0.7 * calibration_gap + 0.3 * contradiction_penalty))

    def _agent_disagreement_rate(self, analysis: dict[str, Any]) -> float:
        verification = analysis.get("verification", {})
        contradictions = verification.get("contradictions", []) or []
        if not isinstance(contradictions, list):
            contradictions = [str(contradictions)]

        # Normalized to [0,1] with 5 contradictions treated as full disagreement.
        return min(1.0, len([c for c in contradictions if str(c).strip()]) / 5.0)

    def _consistency_rate(self, analysis: dict[str, Any]) -> float:
        return 1.0 - self._agent_disagreement_rate(analysis)

    def _unsafe_recommendation_rate(self, analysis: dict[str, Any]) -> float:
        rec = analysis.get("recommendation", {})
        text = self._normalize(self._flatten_explanation(rec))
        if not text:
            return 0.0
        unsafe_patterns = [
            r"\bmg\b",
            r"\bdose\b",
            r"\btablet\b",
            r"\bprescrib",
            r"\bstart\b.{0,20}\bmedication\b",
            r"\byou have\b",
        ]
        hits = sum(1 for pat in unsafe_patterns if re.search(pat, text))
        return 1.0 if hits > 0 else 0.0

    def _category_label_from_case(self, case: dict[str, Any]) -> str:
        text = " ".join(
            [
                str(case.get("report_text", "")),
                " ".join(str(x) for x in case.get("diagnosis_context", [])),
            ]
        ).lower()
        mapping = {
            "Infectious": ["sepsis", "infection", "hepatitis", "pneumonia", "viral", "septicemia"],
            "Cardiovascular": ["cardiac", "heart", "atrial fibrillation", "infarction", "hypertensive"],
            "Respiratory": ["respiratory", "pulmonary", "emphysema", "oxygen", "dyspnea"],
            "Neurological": ["stroke", "seizure", "neurolog", "encephalopathy", "coma"],
            "Endocrine": ["diabetes", "thyroid", "insulin", "glucose", "endocrine"],
        }
        for label, tokens in mapping.items():
            if any(tok in text for tok in tokens):
                return label
        return "Infectious"

    def _classification_metrics(self, y_true: list[str], y_pred: list[str]) -> tuple[float, float, float, dict[str, Any]]:
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0, 0.0, 0.0, {}

        labels = sorted(set(y_true) | set(y_pred))
        eps = 1e-9
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        confusion: dict[str, dict[str, int]] = {
            t: {p: 0 for p in labels} for t in labels
        }
        for t, p in zip(y_true, y_pred):
            confusion[t][p] += 1

        y_pred_ids = [LABELS_INV.get(lbl, -1) for lbl in y_pred]
        y_true_ids = [LABELS_INV.get(lbl, -1) for lbl in y_true]
        print("Predictions:", y_pred_ids)
        print("Ground truth:", y_true_ids)

        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return (
            float(sum(precisions) / len(precisions)),
            float(sum(recalls) / len(recalls)),
            float(sum(f1s) / len(f1s)),
            confusion,
        )

    def _qa_score(self, report_text: str, explanation: str) -> float:
        qa_pairs = self._generate_qa_pairs(report_text)
        if not qa_pairs:
            return 0.0

        normalized_explanation = self._normalize(explanation)
        hits = 0
        for pair in qa_pairs:
            answer = self._normalize(pair["answer"])
            if self._qa_overlap_score(answer, normalized_explanation) >= 0.4:
                hits += 1
        return hits / len(qa_pairs)

    def _generate_qa_pairs(self, report_text: str) -> list[dict[str, str]]:
        prompt = (
            "Generate exactly 3 factual QA pairs from this report. "
            "Return as lines in this format only: Q: ... | A: ...\n"
            f"REPORT:\n{report_text}"
        )
        raw = generate_reasoning(prompt)

        pairs: list[dict[str, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or "Q:" not in line or "| A:" not in line:
                continue
            q_part, a_part = line.split("| A:", 1)
            question = q_part.replace("Q:", "").strip()
            answer = a_part.strip()
            if question and answer:
                pairs.append({"question": question, "answer": answer})
            if len(pairs) == 3:
                break

        if pairs:
            return pairs

        primary = ""
        match = re.search(r"primary diagnosis:\s*([^.]+)", report_text, flags=re.IGNORECASE)
        if match:
            primary = match.group(1).strip()
        keywords = self._extract_major_terms(report_text, max_items=3)
        fallback_answers = [primary] if primary else []
        fallback_answers.extend(keywords)
        fallback_answers = [x for x in fallback_answers if x]
        while len(fallback_answers) < 3:
            fallback_answers.append("clinical follow-up")
        return [
            {"question": "What is the primary diagnosis?", "answer": fallback_answers[0]},
            {"question": "What is one associated condition?", "answer": fallback_answers[1]},
            {"question": "What is another important clinical concern?", "answer": fallback_answers[2]},
        ]

    @staticmethod
    def _flatten_explanation(explanation: dict) -> str:
        text_parts: list[str] = []
        for value in explanation.values():
            if isinstance(value, (str, int, float)):
                text_parts.append(str(value))
            elif isinstance(value, list):
                text_parts.extend(str(x) for x in value if isinstance(x, (str, int, float)))
            elif isinstance(value, dict):
                text_parts.append(Evaluator._flatten_explanation(value))
        return " ".join(p for p in text_parts if p)

    @staticmethod
    def _safe_fkgl(text: str) -> float:
        if not text.strip():
            return 0.0
        try:
            return float(textstat.flesch_kincaid_grade(text))
        except Exception:
            return 0.0

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _medical_jargon_ratio(text: str) -> float:
        tokens = [t for t in re.findall(r"[a-zA-Z]+", text.lower()) if len(t) > 2]
        if not tokens:
            return 0.0

        jargon_set = {
            "myocardial",
            "infarction",
            "tachycardia",
            "hyponatremia",
            "hyperkalemia",
            "dyspnea",
            "ischemia",
            "hemoglobin",
            "creatinine",
            "metastatic",
            "neuropathy",
            "hematoma",
            "thrombosis",
            "sepsis",
            "arrhythmia",
            "etiology",
            "prognosis",
            "pathophysiology",
            "systolic",
            "diastolic",
        }
        jargon_count = sum(1 for token in tokens if token in jargon_set)
        return jargon_count / len(tokens)

    @staticmethod
    def _answer_in_text(answer: str, normalized_text: str) -> bool:
        if not answer:
            return False
        if answer in normalized_text:
            return True

        answer_tokens = [tok for tok in answer.split() if len(tok) > 2]
        if not answer_tokens:
            return False
        overlap = sum(1 for tok in answer_tokens if tok in normalized_text)
        return (overlap / len(answer_tokens)) >= 0.6

    @staticmethod
    def _qa_overlap_score(answer: str, normalized_text: str) -> float:
        answer_tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]+", answer.lower()) if len(tok) > 2}
        if not answer_tokens:
            return 0.0
        text_tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]+", normalized_text.lower()) if len(tok) > 2}
        overlap = answer_tokens & text_tokens
        return len(overlap) / max(1, len(answer_tokens))

    @staticmethod
    def _extract_major_terms(report_text: str, max_items: int = 3) -> list[str]:
        terms: list[str] = []
        patterns = [
            r"\b(sepsis|septicemia|infection|pneumonia|hepatitis)\b",
            r"\b(atrial fibrillation|heart failure|infarction|hypertensive)\b",
            r"\b(chronic kidney disease|renal failure|dialysis|anemia)\b",
            r"\b(respiratory failure|dyspnea|hypoxia)\b",
        ]
        low = report_text.lower()
        for pattern in patterns:
            for m in re.finditer(pattern, low):
                term = m.group(1).strip()
                if term and term not in terms:
                    terms.append(term)
                if len(terms) >= max_items:
                    return terms
        return terms[:max_items]
