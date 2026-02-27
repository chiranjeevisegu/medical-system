from __future__ import annotations

from backend.model_loader import generate_reasoning


class BaselineSummarizer:
    """Single-prompt baseline using FLAN-T5 reasoning model."""

    def summarize(self, report_text: str) -> dict:
        prompt = (
            "Summarize this discharge report in plain language. "
            "Keep important findings and risk context concise.\n"
            f"REPORT:\n{report_text}"
        )
        return {"summary": generate_reasoning(prompt).strip()}
