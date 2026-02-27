from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DATASET_DIR = ROOT_DIR / "dataset"
DATA_DIR = ROOT_DIR / "data"

MODEL_REASONING = "google/flan-t5-large"
MODEL_BIOMEDICAL = "emilyalsentzer/Bio_ClinicalBERT"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    max_cases: int = 20
    dataset_dir: Path = DATASET_DIR
    samples_path: Path = DATA_DIR / "mimic_samples.json"
    results_path: Path = BACKEND_DIR / "results.csv"
    model_local_files_only: bool = _env_bool("MODEL_LOCAL_FILES_ONLY", True)
    hf_token: str = os.getenv("HF_TOKEN", "").strip()
    strict_startup_model_check: bool = _env_bool("STRICT_STARTUP_MODEL_CHECK", True)
    strict_agent_validation: bool = _env_bool("STRICT_AGENT_VALIDATION", True)
    allow_biomedical_fallback: bool = _env_bool("ALLOW_BIOMEDICAL_FALLBACK", False)
    use_gpu_if_available: bool = _env_bool("USE_GPU_IF_AVAILABLE", True)
    use_fp16_on_gpu: bool = _env_bool("USE_FP16_ON_GPU", True)
    classifier_epochs: int = int(os.getenv("CLASSIFIER_EPOCHS", "6"))
    classifier_force_retrain: bool = _env_bool("CLASSIFIER_FORCE_RETRAIN", False)
    classifier_batch_size: int = int(os.getenv("CLASSIFIER_BATCH_SIZE", "8"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
