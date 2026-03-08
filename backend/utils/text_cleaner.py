"""backend/utils/text_cleaner.py

Pre-processing utilities to clean report text before it is passed to agents.

Primary jobs:
1. Strip MIMIC-III de-identification tokens like [**Known lastname 742**]
2. Normalize whitespace
"""
from __future__ import annotations

import re

# MIMIC-III anonymization tokens: [**...**]
_MIMIC_TOKEN_RE = re.compile(r"\[\*\*[^\]]*?\*\*\]")

# Replace common anonymized entities with generic readable text
_MIMIC_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\[\*\*.*?name.*?\*\*\]", re.IGNORECASE),  "the patient"),
    (re.compile(r"\[\*\*.*?date.*?\*\*\]",  re.IGNORECASE),  "the specified date"),
    (re.compile(r"\[\*\*.*?year.*?\*\*\]",  re.IGNORECASE),  "the recorded year"),
    (re.compile(r"\[\*\*.*?age.*?\*\*\]",   re.IGNORECASE),  "their age"),
    (re.compile(r"\[\*\*.*?hospital.*?\*\*\]", re.IGNORECASE), "the treating hospital"),
    (re.compile(r"\[\*\*.*?doctor.*?\*\*\]",   re.IGNORECASE), "the treating physician"),
    (re.compile(r"\[\*\*.*?physician.*?\*\*\]", re.IGNORECASE), "the treating physician"),
    (re.compile(r"\[\*\*.*?location.*?\*\*\]",  re.IGNORECASE), "the clinic"),
    (re.compile(r"\[\*\*.*?company.*?\*\*\]",   re.IGNORECASE), "the facility"),
    (re.compile(r"\[\*\*.*?country.*?\*\*\]",   re.IGNORECASE), "the country"),
    (re.compile(r"\[\*\*.*?state.*?\*\*\]",     re.IGNORECASE), "the state"),
    (re.compile(r"\[\*\*.*?number.*?\*\*\]",    re.IGNORECASE), "the reference number"),
    (re.compile(r"\[\*\*.*?numeric.*?\*\*\]",   re.IGNORECASE), "a numeric value"),
    (re.compile(r"\[\*\*.*?address.*?\*\*\]",   re.IGNORECASE), "the recorded address"),
    (re.compile(r"\[\*\*.*?telephone.*?\*\*\]",  re.IGNORECASE), "the contact number"),
]


def clean_report_text(text: str) -> str:
    """Strip MIMIC-III anonymization tokens and normalise whitespace.

    Applies semantic replacements first (e.g. [**Known lastname 742**] →
    'the patient') then removes any remaining [**...**] tokens.
    """
    for pattern, replacement in _MIMIC_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Any remaining [**...**] tokens → remove entirely
    text = _MIMIC_TOKEN_RE.sub("", text)

    # Collapse excessive whitespace / blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def clean_list_items(items: list[str]) -> list[str]:
    """Clean MIMIC tokens from a list of strings (e.g. agent output items)."""
    return [clean_report_text(item) for item in items if item]
