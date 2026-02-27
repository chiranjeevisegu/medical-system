# Medical Agentic System Runbook

## 1. Prerequisites

- Python 3.12
- Local Hugging Face model cache already downloaded:
  - `google/flan-t5-large`
  - `microsoft/biogpt`

## 2. Install Dependencies

```powershell
python -m pip install -r backend/requirements.txt
```

## 3. Environment Mode

Use offline mode (recommended after model download):

```powershell
$env:MODEL_LOCAL_FILES_ONLY="true"
```

If you need to download missing models:

```powershell
$env:MODEL_LOCAL_FILES_ONLY="false"
```

## 4. Run API

From project root:

```powershell
uvicorn backend.main:app --reload
```

Base URL:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

## 5. Endpoint Execution Order

1. `GET /cases`
2. `POST /analyze/{case_id}`
3. `POST /evaluate`

## 6. Expected Outputs

### `GET /cases`
- Returns list of case IDs.

### `POST /analyze/{case_id}`
- Returns multi-agent output:
  - `clinical_summary`
  - `risk`
  - `temporal_progression`
  - `explanation`
  - `recommendation`
  - `justification`
  - `verification`
  - `user_visible`

### `POST /evaluate`
- Returns aggregate metrics and writes CSV:
  - `backend/results.csv`
- Metrics include:
  - FKGL/readability deltas
  - length reduction
  - QA score
  - patient comprehension
  - information coverage
  - trust calibration
  - disagreement rate

## 7. Current Scale Setting

`backend/config.py` currently uses:

- `max_cases: 5`

For final runs, change to:

- `max_cases: 20`

Then restart API and run `POST /evaluate` again.

## 8. Data Sources Used

The pipeline uses all three:

- `dataset/NOTEEVENTS.csv` (discharge summaries)
- `dataset/DIAGNOSES_ICD.csv` (diagnosis context)
- `dataset/LABEVENTS.csv` (lab context)

## 9. Common Issues

### Model cache error
Symptom: model unavailable in local cache.

Fix:
1. set `MODEL_LOCAL_FILES_ONLY=false`
2. run once to download models
3. switch back to `true`

### Slow evaluation on CPU
Symptom: long runtime.

Fix:
1. keep `max_cases=5` for smoke tests
2. use `max_cases=20` only for final evaluation

### Non-JSON agent outputs
Handled in code with safe parse fallbacks; pipeline should continue.

## 10. Final Demo Script

1. Install deps
2. Set offline mode
3. Start Uvicorn
4. Run `/cases`
5. Run one `/analyze/{case_id}`
6. Run `/evaluate`
7. Open `backend/results.csv` and report metrics
