"""Additional evaluation metrics for the medical agentic system.

Provides semantic similarity scoring using BioClinicalBERT embeddings
as a higher-quality alternative to pure token-overlap QA evaluation.

Usage:
    from backend.utils.metrics import semantic_similarity, semantic_qa_score
"""
from __future__ import annotations

import re


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts via BioClinicalBERT CLS embeddings.

    Returns a float in [0.0, 1.0]. Falls back to 0.0 if embeddings fail.
    """
    # Import here to avoid circular imports and defer model loading.
    try:
        from backend.model_loader import _embed_texts  # noqa: PLC0415
        emb = _embed_texts([str(text_a)[:512], str(text_b)[:512]])
        # emb is already L2-normalized; dot product == cosine similarity.
        score = float((emb[0] @ emb[1]).item())
        return max(0.0, min(1.0, score))
    except Exception:  # noqa: BLE001
        return 0.0


def semantic_qa_score(report_text: str, explanation: str, qa_pairs: list[dict] | None = None) -> float:
    """Compute QA preservation score using semantic similarity instead of token overlap.

    For each QA pair, computes embedding cosine similarity between the
    expected answer and the explanation text. Returns the mean similarity.

    Args:
        report_text: Original clinical report (used to generate QA pairs if not provided).
        explanation: Agent-generated patient-facing explanation.
        qa_pairs: Pre-generated QA pairs. If None, falls back to keyword extraction.

    Returns:
        Float in [0.0, 1.0] representing average semantic similarity.
    """
    if not explanation or not explanation.strip():
        return 0.0

    pairs = qa_pairs or []
    if not pairs:
        # Lightweight fallback: extract clinical keywords as "expected answers"
        pairs = _extract_keyword_pairs(report_text)

    if not pairs:
        return 0.0

    scores: list[float] = []
    for pair in pairs:
        answer = str(pair.get("answer", "")).strip()
        if not answer:
            continue
        sim = semantic_similarity(answer, explanation)
        scores.append(sim)

    return sum(scores) / len(scores) if scores else 0.0


def _extract_keyword_pairs(report_text: str) -> list[dict]:
    """Extract simple keyword-based fallback QA pairs from report text."""
    patterns = [
        r"\b(sepsis|septicemia|infection|pneumonia|hepatitis)\b",
        r"\b(atrial fibrillation|heart failure|infarction|cardiac)\b",
        r"\b(chronic kidney disease|renal failure|dialysis)\b",
        r"\b(respiratory failure|dyspnea|hypoxia|pulmonary)\b",
        r"\b(diabetes|insulin|thyroid|glucose)\b",
        r"\b(stroke|seizure|encephalopathy|confusion)\b",
    ]
    terms: list[str] = []
    low = report_text.lower()
    for pattern in patterns:
        for m in re.finditer(pattern, low):
            term = m.group(1).strip()
            if term and term not in terms:
                terms.append(term)
            if len(terms) >= 3:
                break
        if len(terms) >= 3:
            break

    primary_match = re.search(r"primary diagnosis:\s*([^.]+)", report_text, re.IGNORECASE)
    if primary_match:
        primary = primary_match.group(1).strip()
        if primary and primary.lower() not in terms:
            terms.insert(0, primary)

    while len(terms) < 2:
        terms.append("clinical follow-up required")

    return [{"question": f"What is {t}?", "answer": t} for t in terms[:3]]
