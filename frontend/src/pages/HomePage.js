import React, { useState } from 'react';
import ReportInput from '../components/ReportInput';
import ResultDisplay from '../components/ResultDisplay';
import HistoryPanel, { saveToHistory } from '../components/HistoryPanel';
import './HomePage.css';

const BACKEND = 'http://localhost:8000';

export default function HomePage() {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [historyKey, setHistoryKey] = useState(0);

    const handleAnalyze = async (reportText) => {
        setLoading(true);
        setResult(null);
        setError(null);
        try {
            const res = await fetch(`${BACKEND}/analyze_report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ report_text: reportText }),
            });
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data?.detail || `Server error ${res.status}`);
            }
            const data = await res.json();
            setResult(data);
            saveToHistory(reportText, data);
            setHistoryKey(k => k + 1);  // trigger HistoryPanel refresh
            setTimeout(() => {
                document.getElementById('results-anchor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        } catch (err) {
            setError(err.message || 'Network error. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    const handleSelectHistory = (historicResult) => {
        setResult(historicResult);
        setError(null);
        setTimeout(() => {
            document.getElementById('results-anchor')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    };

    return (
        <main>
            {/* Hero */}
            <div className="hero">
                <div className="hero-glow" />
                <div className="page-container">
                    <div className="hero-content">
                        <div className="hero-badge badge badge-info">
                            <span className="hero-badge-dot" />
                            Multi-Agent Clinical NLP System
                        </div>
                        <h1 className="hero-title">
                            AI Medical Report<br />
                            <span className="gradient-text">Understanding System</span>
                        </h1>
                        <p className="hero-sub">
                            Powered by <strong>BioClinicalBERT</strong> + <strong>FLAN-T5</strong> seven-agent pipeline.
                            Converts complex clinical discharge summaries into structured, patient-friendly explanations.
                        </p>
                        <div className="hero-stats">
                            {[
                                { label: 'AI Agents', value: '6' },
                                { label: 'NLP Models', value: '2' },
                                { label: 'Output Sections', value: '7' },
                                { label: 'Dataset', value: 'MIMIC-III' },
                            ].map(s => (
                                <div key={s.label} className="hero-stat">
                                    <span className="stat-value gradient-text">{s.value}</span>
                                    <span className="stat-label">{s.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Main content */}
            <div className="page-container home-content">
                <ReportInput onSubmit={handleAnalyze} loading={loading} />
                <HistoryPanel onSelect={handleSelectHistory} refreshKey={historyKey} />

                <div id="results-anchor" />
                <ResultDisplay result={result} error={error} loading={loading} />
            </div>
        </main>
    );
}
