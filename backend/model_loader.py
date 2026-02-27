from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from backend.config import BACKEND_DIR, MODEL_BIOMEDICAL, MODEL_REASONING, get_settings


# Loaded lazily on first use to avoid import-time crashes when caches are missing.
_REASONING_TOKENIZER = None
_REASONING_MODEL = None
_BIOMEDICAL_TOKENIZER = None
_BIOMEDICAL_MODEL = None
_CLINICAL_CLASSIFIER = None
_CLASSIFIER_DEVICE = None

CATEGORY_LABELS = [
    "Cardiovascular",
    "Respiratory",
    "Endocrine",
    "Infectious",
    "Neurological",
]

_CATEGORY_TO_IDX = {name: i for i, name in enumerate(CATEGORY_LABELS)}
_CLASSIFIER_PATH = BACKEND_DIR / "clinical_classifier.pt"


def _should_use_gpu() -> bool:
    settings = get_settings()
    return settings.use_gpu_if_available and torch.cuda.is_available()


def _preferred_dtype():
    settings = get_settings()
    if _should_use_gpu() and settings.use_fp16_on_gpu:
        return torch.float16
    return torch.float32


def _first_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if _should_use_gpu() else "cpu")


def _load_reasoning_once():
    global _REASONING_TOKENIZER, _REASONING_MODEL
    if _REASONING_TOKENIZER is not None and _REASONING_MODEL is not None:
        return _REASONING_TOKENIZER, _REASONING_MODEL
    settings = get_settings()
    common_kwargs = {
        "local_files_only": settings.model_local_files_only,
    }
    if settings.hf_token:
        common_kwargs["token"] = settings.hf_token
    try:
        _REASONING_TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_REASONING, **common_kwargs
        )
        if _should_use_gpu():
            try:
                _REASONING_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_REASONING,
                    dtype=_preferred_dtype(),
                    device_map="auto",
                    **common_kwargs,
                )
            except Exception:
                _REASONING_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_REASONING,
                    device_map="auto",
                    **common_kwargs,
                )
        else:
            _REASONING_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_REASONING,
                **common_kwargs,
            )
    except Exception as exc:
        raise RuntimeError(
            "Reasoning model is unavailable in local cache. "
            f"Expected model: {MODEL_REASONING}. "
            "Set MODEL_LOCAL_FILES_ONLY=false to allow download, "
            "and optionally set HF_TOKEN for gated models."
        ) from exc
    return _REASONING_TOKENIZER, _REASONING_MODEL


def _load_biomedical_once():
    global _BIOMEDICAL_TOKENIZER, _BIOMEDICAL_MODEL
    if _BIOMEDICAL_TOKENIZER is not None and _BIOMEDICAL_MODEL is not None:
        return _BIOMEDICAL_TOKENIZER, _BIOMEDICAL_MODEL
    settings = get_settings()
    common_kwargs = {
        "local_files_only": settings.model_local_files_only,
    }
    if settings.hf_token:
        common_kwargs["token"] = settings.hf_token
    bio_kwargs = dict(common_kwargs)
    bio_kwargs["use_safetensors"] = True
    try:
        _BIOMEDICAL_TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_BIOMEDICAL, **common_kwargs
        )
        if _should_use_gpu():
            try:
                _BIOMEDICAL_MODEL = AutoModel.from_pretrained(
                    MODEL_BIOMEDICAL,
                    dtype=_preferred_dtype(),
                    **bio_kwargs,
                )
            except Exception:
                _BIOMEDICAL_MODEL = AutoModel.from_pretrained(
                    MODEL_BIOMEDICAL,
                    **bio_kwargs,
                )
            _BIOMEDICAL_MODEL = _BIOMEDICAL_MODEL.to("cuda")
        else:
            _BIOMEDICAL_MODEL = AutoModel.from_pretrained(
                MODEL_BIOMEDICAL,
                **bio_kwargs,
            )
    except Exception as exc:
        msg = str(exc)
        if "CVE-2025-32434" in msg or "upgrade torch to at least v2.6" in msg:
            raise RuntimeError(
                "BioClinicalBERT load blocked by torch security policy. "
                "Install torch>=2.6.0 and restart."
            ) from exc
        raise RuntimeError(
            "Biomedical model is unavailable in local cache. "
            f"Expected model: {MODEL_BIOMEDICAL}. "
            "Set MODEL_LOCAL_FILES_ONLY=false to allow download, "
            "and optionally set HF_TOKEN for gated models."
        ) from exc
    return _BIOMEDICAL_TOKENIZER, _BIOMEDICAL_MODEL


def _generate_text(tokenizer, model, prompt: str, *, max_new_tokens: int, temperature: float, do_sample: bool, top_p: float | None = None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = _first_model_device(model)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    torch.manual_seed(42)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    if top_p is not None and do_sample:
        kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**kwargs)

    if getattr(model.config, "is_encoder_decoder", False):
        generated_ids = output_ids[0]
    else:
        generated_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_reasoning(prompt: str) -> str:
    tokenizer, model = _load_reasoning_once()
    return _generate_text(
        tokenizer,
        model,
        prompt,
        max_new_tokens=96,
        temperature=0.0,
        top_p=None,
        do_sample=False,
    )


def generate_biomedical(prompt: str) -> str:
    # Compatibility shim: biomedical extraction now uses BioClinicalBERT embeddings.
    return generate_reasoning(prompt)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _embed_texts(texts: list[str]) -> torch.Tensor:
    tokenizer, model = _load_biomedical_once()
    device = _first_model_device(model)
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return F.normalize(cls_embeddings, p=2, dim=1)


class ClinicalClassifierHead(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = len(CATEGORY_LABELS)) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def _infer_category_label(text: str) -> str:
    t = text.lower()
    rules = {
        "Infectious": [
            "sepsis",
            "septicemia",
            "infection",
            "hepatitis",
            "pneumonia",
            "viral",
            "bacterial",
        ],
        "Cardiovascular": [
            "heart",
            "cardiac",
            "atrial fibrillation",
            "infarction",
            "hypertensive",
            "cardiogenic",
        ],
        "Respiratory": [
            "respiratory",
            "pulmonary",
            "dyspnea",
            "oxygen",
            "emphysema",
            "airway",
        ],
        "Neurological": [
            "stroke",
            "seizure",
            "neurolog",
            "encephalopathy",
            "confusion",
            "coma",
        ],
        "Endocrine": [
            "diabetes",
            "thyroid",
            "insulin",
            "glucose",
            "adrenal",
            "endocrine",
        ],
    }
    priority = ["Infectious", "Cardiovascular", "Respiratory", "Endocrine", "Neurological"]
    for label in priority:
        for token in rules[label]:
            if token in t:
                return label
    return "Infectious"


def _build_classifier_training_data(samples: list[dict]) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for sample in samples:
        report_text = str(sample.get("report_text", "")).strip()
        diag_ctx = sample.get("diagnosis_context", [])
        joined = " ".join([report_text] + [str(x) for x in diag_ctx if str(x).strip()])
        if not joined:
            continue
        label = _infer_category_label(joined)
        texts.append(joined[:4000])
        labels.append(_CATEGORY_TO_IDX[label])
    return texts, labels


def train_clinical_classifier(samples: list[dict], epochs: int = 6, force_retrain: bool = False) -> dict[str, float | int]:
    global _CLINICAL_CLASSIFIER, _CLASSIFIER_DEVICE
    if _CLASSIFIER_PATH.exists() and not force_retrain and _CLINICAL_CLASSIFIER is not None:
        return {"epochs": 0, "train_loss": 0.0}

    texts, labels = _build_classifier_training_data(samples)
    if not texts:
        raise RuntimeError("No training data available for clinical classifier head.")

    x = _embed_texts(texts).float()
    y = torch.tensor(labels, dtype=torch.long, device=x.device)
    _CLASSIFIER_DEVICE = x.device

    model = ClinicalClassifierHead(input_dim=x.shape[-1], num_classes=len(CATEGORY_LABELS)).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    counts = torch.bincount(y, minlength=len(CATEGORY_LABELS)).float()
    class_weights = torch.ones(len(CATEGORY_LABELS), dtype=torch.float32, device=x.device)
    nonzero = counts > 0
    if torch.any(nonzero):
        class_weights[nonzero] = float(len(labels)) / (len(CATEGORY_LABELS) * counts[nonzero])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.train()

    last_loss = 0.0
    settings = get_settings()
    batch_size = max(8, min(16, int(settings.classifier_batch_size)))
    epochs_eff = int(max(5, min(10, epochs)))
    num_items = x.shape[0]
    for _ in range(epochs_eff):
        perm = torch.randperm(num_items, device=x.device)
        running = 0.0
        steps = 0
        for start in range(0, num_items, batch_size):
            idx = perm[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        last_loss = running / max(steps, 1)

    _CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": CATEGORY_LABELS,
            "meta": {"epochs": epochs_eff, "train_size": len(texts), "batch_size": batch_size},
        },
        _CLASSIFIER_PATH,
    )
    _CLINICAL_CLASSIFIER = model.eval()
    return {"epochs": epochs_eff, "train_loss": last_loss, "train_size": len(texts), "batch_size": batch_size}


def _load_classifier_if_exists() -> bool:
    global _CLINICAL_CLASSIFIER, _CLASSIFIER_DEVICE
    if _CLINICAL_CLASSIFIER is not None:
        return True
    if not _CLASSIFIER_PATH.exists():
        return False
    checkpoint = torch.load(_CLASSIFIER_PATH, map_location="cpu", weights_only=True)
    model = ClinicalClassifierHead(input_dim=768, num_classes=len(CATEGORY_LABELS))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda:0" if _should_use_gpu() else "cpu")
    _CLINICAL_CLASSIFIER = model.to(device).eval()
    _CLASSIFIER_DEVICE = device
    return True


def ensure_clinical_classifier(samples: list[dict]) -> dict[str, float | int | str]:
    settings = get_settings()
    if _load_classifier_if_exists() and not settings.classifier_force_retrain:
        return {"status": "loaded", "path": str(_CLASSIFIER_PATH)}
    stats = train_clinical_classifier(
        samples=samples,
        epochs=settings.classifier_epochs,
        force_retrain=settings.classifier_force_retrain,
    )
    return {"status": "trained", "path": str(_CLASSIFIER_PATH), **stats}


def predict_disease_category(report_text: str) -> dict[str, float | str]:
    if not _load_classifier_if_exists():
        raise RuntimeError("Clinical classifier head is not available. Train or initialize it first.")
    text = report_text.strip()
    if not text:
        return {"disease_category": "Infectious", "disease_confidence": 0.0}
    emb = _embed_texts([text[:4000]]).float()
    model = _CLINICAL_CLASSIFIER
    if model is None:
        raise RuntimeError("Clinical classifier head failed to load.")
    with torch.inference_mode():
        logits = model(emb)
        probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())
    predicted = CATEGORY_LABELS[idx]
    if conf < 0.4:
        predicted = _infer_category_label(text)
    return {"disease_category": predicted, "disease_confidence": conf}


def _clean_clinical_items(items: list[str]) -> list[str]:
    blocked_exact = {"DIAGNOSES_ICD_CONTEXT", "LABEVENTS_CONTEXT", "DISCHARGE_SUMMARY"}
    blocked_contains = ["_context", "context:"]
    out: list[str] = []
    for raw in items:
        s = re.sub(r"\s+", " ", str(raw or "")).strip(" -:\t")
        if not s:
            continue
        if s.upper() in blocked_exact:
            continue
        if any(b in s.lower() for b in blocked_contains):
            continue
        s = re.sub(r"^(primary diagnosis:\s*)", "", s, flags=re.IGNORECASE).strip()
        if s.lower().startswith("associated icd diagnoses"):
            continue
        if len(s) < 3:
            continue
        if s not in out:
            out.append(s)
    return out[:5]


def extract_clinical_facts(report_text: str) -> dict[str, list[str]]:
    text = report_text.strip()
    if not text:
        return {"diagnosis": [], "abnormal_findings": [], "key_observations": []}

    sentences = [
        s.strip(" -:\t")
        for s in re.split(r"[.\n;]+", text)
        if s and len(s.strip()) > 20
    ][:80]
    if not sentences:
        return {"diagnosis": [], "abnormal_findings": [], "key_observations": []}

    sentence_emb = _embed_texts(sentences)
    query_texts = [
        "diagnosis impression assessment disease condition",
        "abnormal elevated low decreased increased critical severe abnormality",
        "admission history hospital course discharge follow up plan",
    ]
    query_emb = _embed_texts(query_texts)
    sim = sentence_emb @ query_emb.T

    def top_for(col: int, k: int = 4) -> list[str]:
        values, idx = torch.topk(sim[:, col], k=min(k, sim.shape[0]))
        out: list[str] = []
        for score, i in zip(values.tolist(), idx.tolist()):
            if score < 0.05:
                continue
            sent = sentences[i].strip()
            if sent and sent not in out:
                out.append(sent)
        return out

    diagnosis = _clean_clinical_items(top_for(0, 4))
    abnormal = _clean_clinical_items(top_for(1, 4))
    observations = _clean_clinical_items(top_for(2, 5))

    primary = re.search(r"primary diagnosis:\s*([^.]+)", text, flags=re.IGNORECASE)
    if primary:
        p = re.sub(r"\s+", " ", primary.group(1)).strip(" -:\t")
        if p and p not in diagnosis:
            diagnosis.insert(0, p)

    for sent in sentences:
        low = sent.lower()
        if any(k in low for k in ["sepsis", "shock", "failure", "fracture", "hepatitis", "infarction"]):
            if sent not in diagnosis and len(diagnosis) < 5:
                diagnosis.append(sent)
        if any(k in low for k in ["elevated", "decreased", "low", "high", "critical", "abnormal"]):
            if sent not in abnormal and len(abnormal) < 5:
                abnormal.append(sent)

    return {
        "diagnosis": _clean_clinical_items(diagnosis),
        "abnormal_findings": _clean_clinical_items(abnormal),
        "key_observations": _clean_clinical_items(observations),
    }


def preflight_models() -> dict[str, str]:
    """Fail-fast health check to avoid silent runtime failures."""
    reason_tok, reason_model = _load_reasoning_once()
    _ = _generate_text(
        reason_tok,
        reason_model,
        "Summarize: Patient stable.",
        max_new_tokens=8,
        temperature=0.0,
        do_sample=False,
    )

    _ = _load_biomedical_once()
    _ = extract_clinical_facts("Primary diagnosis: sepsis. Elevated creatinine and low blood pressure.")
    return {
        "reasoning_model": MODEL_REASONING,
        "biomedical_model": MODEL_BIOMEDICAL,
        "cuda_available": str(torch.cuda.is_available()),
        "gpu_enabled": str(_should_use_gpu()),
        "classifier_path": str(_CLASSIFIER_PATH),
    }


def runtime_status(load_models: bool = False) -> dict[str, str]:
    """Report runtime acceleration/model placement state."""
    settings = get_settings()
    cuda_available = torch.cuda.is_available()
    status: dict[str, str] = {
        "reasoning_model": MODEL_REASONING,
        "biomedical_model": MODEL_BIOMEDICAL,
        "cuda_available": str(cuda_available),
        "gpu_enabled_config": str(settings.use_gpu_if_available),
        "gpu_active": str(_should_use_gpu()),
        "reasoning_loaded": str(_REASONING_MODEL is not None),
        "biomedical_loaded": str(_BIOMEDICAL_MODEL is not None),
        "classifier_loaded": str(_CLINICAL_CLASSIFIER is not None),
        "reasoning_device": "not_loaded",
        "biomedical_device": "not_loaded",
        "classifier_device": "not_loaded",
    }

    if cuda_available:
        status["cuda_device_count"] = str(torch.cuda.device_count())
        try:
            status["cuda_device_name_0"] = torch.cuda.get_device_name(0)
        except Exception:
            status["cuda_device_name_0"] = "unknown"

    if load_models:
        _load_reasoning_once()
        _load_biomedical_once()
        _load_classifier_if_exists()
        status["reasoning_loaded"] = str(_REASONING_MODEL is not None)
        status["biomedical_loaded"] = str(_BIOMEDICAL_MODEL is not None)
        status["classifier_loaded"] = str(_CLINICAL_CLASSIFIER is not None)

    if _REASONING_MODEL is not None:
        status["reasoning_device"] = str(_first_model_device(_REASONING_MODEL))
    if _BIOMEDICAL_MODEL is not None:
        status["biomedical_device"] = str(_first_model_device(_BIOMEDICAL_MODEL))
    if _CLINICAL_CLASSIFIER is not None and _CLASSIFIER_DEVICE is not None:
        status["classifier_device"] = str(_CLASSIFIER_DEVICE)
    status["classifier_path"] = str(_CLASSIFIER_PATH)
    status["classifier_exists"] = str(_CLASSIFIER_PATH.exists())

    return status
