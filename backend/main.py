from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from backend.baseline import BaselineSummarizer
from backend.config import get_settings
from backend.dataset_loader import MIMICDatasetLoader
from backend.evaluation import Evaluator
from backend.model_loader import preflight_models, runtime_status, train_clinical_classifier
from backend.orchestrator import Orchestrator

app = FastAPI(title="Explainable Multi-Agent Medical Report Understanding System")
_ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}


@app.on_event("startup")
def startup_checks() -> None:
    settings = get_settings()
    if settings.strict_startup_model_check:
        preflight_models()


@lru_cache(maxsize=1)
def get_loader() -> MIMICDatasetLoader:
    settings = get_settings()
    loader = MIMICDatasetLoader(
        dataset_dir=settings.dataset_dir,
        output_path=settings.samples_path,
        max_cases=settings.max_cases,
    )
    loader.prepare_samples(force_refresh=False)
    return loader


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator(get_settings())


@lru_cache(maxsize=1)
def get_baseline() -> BaselineSummarizer:
    return BaselineSummarizer()


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=307)


@app.get("/dashboard/first5")
def dashboard_first5() -> dict[str, Any]:
    try:
        orchestrator = get_orchestrator()
        evaluator = Evaluator(orchestrator, get_baseline(), orchestrator.settings.results_path)

        case_ids = orchestrator.list_case_ids()[:5]
        rows: list[dict[str, Any]] = []
        for case_id in case_ids:
            case = orchestrator.get_case(case_id)
            report_text = str(case.get("report_text", ""))
            # Lightweight local baseline for dashboard speed.
            baseline_summary = " ".join(report_text.split()[:120])
            analysis = orchestrator.analyze_case(case_id)
            agent_explanation = evaluator._flatten_explanation(analysis.get("explanation", {}))
            information_coverage = evaluator._information_coverage(analysis, agent_explanation)

            rows.append(
                {
                    "case_id": case_id,
                    "analysis": analysis,
                    "metrics": {
                        "original_fkgl": evaluator._safe_fkgl(report_text),
                        "baseline_fkgl": evaluator._safe_fkgl(baseline_summary),
                        "agent_fkgl": evaluator._safe_fkgl(agent_explanation),
                        "original_length": evaluator._word_count(report_text),
                        "baseline_length": evaluator._word_count(baseline_summary),
                        "agent_length": evaluator._word_count(agent_explanation),
                        "qa_score": information_coverage,
                        "patient_comprehension": evaluator._patient_comprehension_score(agent_explanation),
                        "information_coverage": information_coverage,
                        "trust_calibration": evaluator._trust_calibration_score(analysis),
                        "agent_disagreement_rate": evaluator._agent_disagreement_rate(analysis),
                    },
                }
            )
        return {"count": len(rows), "cases": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health(load_models: bool = False) -> dict[str, Any]:
    try:
        return {
            "status": "ok",
            "runtime": runtime_status(load_models=load_models),
            "endpoints": ["/cases", "/analyze/{case_id}", "/metrics/{case_id}", "/train_classifier", "/evaluate?limit=5", "/app"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/{case_id}")
def metrics_for_case(case_id: str) -> dict[str, Any]:
    try:
        orchestrator = get_orchestrator()
        evaluator = Evaluator(orchestrator, get_baseline(), orchestrator.settings.results_path)
        case = orchestrator.get_case(case_id)
        report_text = str(case.get("report_text", ""))
        baseline_summary = " ".join(report_text.split()[:120])

        analysis = _ANALYSIS_CACHE.get(case_id)
        if analysis is None:
            analysis = orchestrator.analyze_case(case_id)
            _ANALYSIS_CACHE[case_id] = analysis

        agent_explanation = evaluator._flatten_explanation(analysis.get("explanation", {}))
        information_coverage = evaluator._information_coverage(analysis, agent_explanation)
        return {
            "case_id": case_id,
            "original_fkgl": evaluator._safe_fkgl(report_text),
            "baseline_fkgl": evaluator._safe_fkgl(baseline_summary),
            "agent_fkgl": evaluator._safe_fkgl(agent_explanation),
            "original_length": evaluator._word_count(report_text),
            "baseline_length": evaluator._word_count(baseline_summary),
            "agent_length": evaluator._word_count(agent_explanation),
            "qa_score": information_coverage,
            "patient_comprehension": evaluator._patient_comprehension_score(agent_explanation),
            "information_coverage": information_coverage,
            "trust_calibration": evaluator._trust_calibration_score(analysis),
            "agent_disagreement_rate": evaluator._agent_disagreement_rate(analysis),
            "consistency_rate": evaluator._consistency_rate(analysis),
            "unsafe_recommendation_rate": evaluator._unsafe_recommendation_rate(analysis),
            "classifier_pred_label": analysis.get("clinical_summary", {}).get("disease_category", ""),
            "classifier_pred_confidence": analysis.get("clinical_summary", {}).get("disease_confidence", 0.0),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train_classifier")
def train_classifier(epochs: int = 6, force_retrain: bool = False) -> dict[str, Any]:
    try:
        orchestrator = get_orchestrator()
        stats = train_clinical_classifier(
            samples=orchestrator.samples,
            epochs=max(5, min(10, int(epochs))),
            force_retrain=bool(force_retrain),
        )
        return {
            "status": "ok",
            "message": "BioClinicalBERT classifier head training complete.",
            "stats": stats,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/app", response_class=HTMLResponse)
def app_ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Medical Agentic System</title>
  <style>
    :root {
      --bg: #f7f7f2;
      --card: #ffffff;
      --ink: #1b1d1f;
      --muted: #5f666d;
      --accent: #146c94;
      --accent-2: #0f4c5c;
      --ok: #1f7a1f;
      --warn: #9a6700;
      --line: #d9dee3;
    }
    body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--ink); }
    .wrap { max-width: 1000px; margin: 24px auto; padding: 0 16px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 14px; }
    h1 { margin: 0 0 12px; font-size: 26px; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button { border: 0; border-radius: 8px; background: var(--accent); color: #fff; padding: 9px 12px; cursor: pointer; }
    button:hover { background: var(--accent-2); }
    input, select { border: 1px solid var(--line); border-radius: 8px; padding: 9px; min-width: 240px; }
    .small { color: var(--muted); font-size: 13px; }
    pre { white-space: pre-wrap; word-wrap: break-word; background: #f8fafc; border: 1px solid var(--line); border-radius: 8px; padding: 12px; max-height: 460px; overflow: auto; }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .report-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .report-card { border: 1px solid var(--line); border-radius: 10px; padding: 12px; background: #fcfdff; }
    .report-title { font-weight: 600; margin-bottom: 6px; }
    .report-meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Medical Agentic System</h1>
    <div class="small">Auto view for first 5 NOTEEVENTS discharge summaries with full outputs and metrics.</div>

    <div class="card">
      <h2>Auto Output: First 5 Reports (NOTEEVENTS)</h2>
      <div id="autoStatus" class="small">Preparing first 5 outputs...</div>
      <div id="autoOut" class="report-grid"></div>
    </div>

    <div class="card">
      <h2>Manual: Load Cases</h2>
      <div class="row">
        <button id="loadCases">GET /cases</button>
        <select id="caseSelect"><option value="">Select case...</option></select>
      </div>
      <div id="caseStatus" class="small"></div>
    </div>

    <div class="card">
      <h2>2) Analyze Case</h2>
      <div class="row">
        <button id="analyzeBtn">POST /analyze/{case_id}</button>
      </div>
      <pre id="analyzeOut">No analysis yet.</pre>
    </div>

    <div class="card">
      <h2>3) Evaluate All</h2>
      <div class="row">
        <button id="evalBtn">POST /evaluate</button>
        <span class="small warn">CPU run can take time.</span>
      </div>
      <pre id="evalOut">No evaluation yet.</pre>
    </div>
  </div>

  <script>
    const caseSelect = document.getElementById("caseSelect");
    const caseStatus = document.getElementById("caseStatus");
    const analyzeOut = document.getElementById("analyzeOut");
    const evalOut = document.getElementById("evalOut");
    const autoStatus = document.getElementById("autoStatus");
    const autoOut = document.getElementById("autoOut");

    async function getJson(url, opts = {}) {
      const res = await fetch(url, opts);
      const text = await res.text();
      let body;
      try { body = JSON.parse(text); } catch { body = text; }
      if (!res.ok) throw new Error(typeof body === "string" ? body : JSON.stringify(body, null, 2));
      return body;
    }

    document.getElementById("loadCases").addEventListener("click", async () => {
      caseStatus.textContent = "Loading cases...";
      try {
        const cases = await getJson("/cases");
        caseSelect.innerHTML = '<option value="">Select case...</option>';
        for (const id of cases) {
          const opt = document.createElement("option");
          opt.value = id;
          opt.textContent = id;
          caseSelect.appendChild(opt);
        }
        caseStatus.innerHTML = `<span class="ok">Loaded ${cases.length} cases.</span>`;
      } catch (err) {
        caseStatus.innerHTML = `<span class="warn">Error: ${err.message}</span>`;
      }
    });

    document.getElementById("analyzeBtn").addEventListener("click", async () => {
      const caseId = caseSelect.value;
      if (!caseId) {
        analyzeOut.textContent = "Pick a case first from the dropdown.";
        return;
      }
      analyzeOut.textContent = "Analyzing... please wait.";
      try {
        const out = await getJson(`/analyze/${encodeURIComponent(caseId)}`, { method: "POST" });
        analyzeOut.textContent = JSON.stringify(out, null, 2);
      } catch (err) {
        analyzeOut.textContent = `Error: ${err.message}`;
      }
    });

    document.getElementById("evalBtn").addEventListener("click", async () => {
      evalOut.textContent = "Evaluating... this can be slow on CPU.";
      try {
        const out = await getJson("/evaluate?limit=20", { method: "POST" });
        evalOut.textContent = JSON.stringify(out, null, 2);
      } catch (err) {
        evalOut.textContent = `Error: ${err.message}`;
      }
    });

    function summarizeOutput(caseId, out, metrics) {
      const explanation = out.explanation || {};
      const recommendation = out.recommendation || {};
      const verification = out.verification || {};
      const risk = (out.risk && out.risk.risk_level) ? out.risk.risk_level : "Unknown";
      const nextSteps = recommendation.safe_next_steps || [];
      const urgent = recommendation.urgent_attention_signs || [];
      return {
        caseId,
        risk,
        simple: explanation.simple_explanation || "",
        takeaway: explanation.plain_language_takeaway || "",
        consistency: verification.consistency_score,
        nextSteps,
        urgent,
        metrics: metrics || {}
      };
    }

    function renderCard(summary) {
      const div = document.createElement("div");
      div.className = "report-card";
      const stepText = (summary.nextSteps || []).map((x) => `- ${x}`).join("\\n");
      const urgentText = (summary.urgent || []).map((x) => `- ${x}`).join("\\n");
      const m = summary.metrics || {};
      const metricsText = [
        `original_fkgl: ${m.original_fkgl ?? "N/A"}`,
        `baseline_fkgl: ${m.baseline_fkgl ?? "N/A"}`,
        `agent_fkgl: ${m.agent_fkgl ?? "N/A"}`,
        `original_length: ${m.original_length ?? "N/A"}`,
        `baseline_length: ${m.baseline_length ?? "N/A"}`,
        `agent_length: ${m.agent_length ?? "N/A"}`,
        `qa_score: ${m.qa_score ?? "N/A"}`,
        `patient_comprehension: ${m.patient_comprehension ?? "N/A"}`,
        `information_coverage: ${m.information_coverage ?? "N/A"}`,
        `trust_calibration: ${m.trust_calibration ?? "N/A"}`,
        `agent_disagreement_rate: ${m.agent_disagreement_rate ?? "N/A"}`,
        `consistency_rate: ${m.consistency_rate ?? "N/A"}`,
        `unsafe_recommendation_rate: ${m.unsafe_recommendation_rate ?? "N/A"}`,
        `classifier_pred_label: ${m.classifier_pred_label ?? "N/A"}`,
        `classifier_pred_confidence: ${m.classifier_pred_confidence ?? "N/A"}`
      ].join("\\n");
      div.innerHTML = `
        <div class="report-title">${summary.caseId}</div>
        <div class="report-meta">Risk: ${summary.risk} | Verification: ${summary.consistency ?? "N/A"}</div>
        <pre>${summary.simple || "No explanation."}</pre>
        <pre>${summary.takeaway || "No patient takeaway."}</pre>
        <pre>Safe Next Steps:\\n${stepText || "- None"}</pre>
        <pre>Urgent Attention Signs:\\n${urgentText || "- None"}</pre>
        <pre>Metrics:\\n${metricsText}</pre>
      `;
      return div;
    }

    async function runFirstFiveAuto() {
      autoOut.innerHTML = "";
      autoStatus.textContent = "Loading first 5 cases and running analysis...";
      try {
        const cases = await getJson("/cases");
        const firstFive = cases.slice(0, 5);
        caseSelect.innerHTML = '<option value="">Select case...</option>';
        for (const id of firstFive) {
          const opt = document.createElement("option");
          opt.value = id;
          opt.textContent = id;
          caseSelect.appendChild(opt);
        }
        caseStatus.innerHTML = `<span class="ok">Loaded ${firstFive.length} cases.</span>`;

        if (!firstFive.length) {
          autoStatus.innerHTML = '<span class="warn">No cases available.</span>';
          return;
        }
        for (let i = 0; i < firstFive.length; i++) {
          const caseId = firstFive[i];
          autoStatus.textContent = `Processing ${i + 1}/${firstFive.length}: ${caseId}`;
          const analysis = await getJson(`/analyze/${encodeURIComponent(caseId)}`, { method: "POST" });
          const metrics = await getJson(`/metrics/${encodeURIComponent(caseId)}`);
          autoOut.appendChild(renderCard(summarizeOutput(caseId, analysis, metrics)));
        }
        autoStatus.innerHTML = `<span class="ok">Loaded outputs + metrics for ${firstFive.length} reports.</span>`;
      } catch (err) {
        autoStatus.innerHTML = `<span class="warn">Error: ${err.message}</span>`;
      }
    }

    runFirstFiveAuto();
  </script>
</body>
</html>"""


@app.get("/cases")
def list_cases() -> list[str]:
    try:
        cases = get_loader().prepare_samples(force_refresh=False)
        return [sample["case_id"] for sample in cases]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze/{case_id}")
def analyze_case(case_id: str) -> dict:
    try:
        result = get_orchestrator().analyze_case(case_id)
        _ANALYSIS_CACHE[case_id] = result
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate(limit: int | None = 20) -> dict:
    try:
        orchestrator = get_orchestrator()
        evaluator = Evaluator(orchestrator, get_baseline(), orchestrator.settings.results_path)
        case_ids = orchestrator.list_case_ids()
        selected = case_ids[: max(1, min(limit, len(case_ids)))] if limit else case_ids[:20]
        return evaluator.evaluate_all(case_ids=selected)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
