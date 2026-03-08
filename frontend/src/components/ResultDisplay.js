import React from 'react';
import './ResultDisplay.css';

const RISK_COLORS = { High: 'danger', Medium: 'warning', Low: 'success' };
const RISK_ICONS = { High: '🔴', Medium: '🟡', Low: '🟢' };

function ConfidenceBar({ value }) {
    const pct = Math.round(value * 100);
    const color = pct >= 80 ? 'var(--success)' : pct >= 60 ? 'var(--warning)' : 'var(--danger)';
    return (
        <div className="conf-bar-wrap">
            <div className="conf-bar-track">
                <div className="conf-bar-fill" style={{ width: `${pct}%`, background: color }} />
            </div>
            <span className="conf-value" style={{ color }}>{pct}%</span>
        </div>
    );
}

function TagList({ items, variant = 'success' }) {
    if (!items || items.length === 0) return <p className="empty-msg">No items returned.</p>;
    return (
        <ul className="tag-list">
            {items.map((item, i) => (
                <li key={i} className={`tag-item tag-${variant}`}>{item}</li>
            ))}
        </ul>
    );
}

export default function ResultDisplay({ result, error, loading }) {
    if (loading) {
        return (
            <div className="result-loading fade-in">
                <div className="loading-inner">
                    <div className="spinner" />
                    <p className="loading-title">Analyzing Report…</p>
                    <p className="loading-sub">Running BioClinicalBERT + FLAN-T5 pipeline through 6 agents</p>
                    <div className="pipeline-dots">
                        {['Clinical Extraction', 'Risk Assessment', 'Explanation', 'Recommendation', 'Justification', 'Verification']
                            .map((s, i) => (
                                <div key={s} className="pipeline-dot" style={{ animationDelay: `${i * 0.3}s` }}>
                                    <span className="dot-pulse" />
                                    <span className="dot-label">{s}</span>
                                </div>
                            ))}
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="result-error section-card fade-in">
                <h2>⚠️ Analysis Failed</h2>
                <p className="error-msg">{error}</p>
                <p className="error-hint">Make sure the backend is running: <code>uvicorn backend.api:app --reload</code></p>
            </div>
        );
    }

    if (!result) return null;

    const { disease_category, disease_confidence, risk, explanation, recommendation, justification, verification, processing_time_seconds } = result;
    const riskColor = RISK_COLORS[risk?.risk_level] || 'info';
    const riskIcon = RISK_ICONS[risk?.risk_level] || '⚪';

    return (
        <div className="results-wrapper fade-in">
            {/* Header Summary */}
            <div className="result-header section-card">
                <div className="result-header-grid">
                    <div className="header-cell">
                        <label>Disease Category</label>
                        <span className="category-tag badge badge-info">{disease_category}</span>
                    </div>
                    <div className="header-cell">
                        <label>Risk Level</label>
                        <span className={`badge badge-${riskColor}`}>{riskIcon} {risk?.risk_level}</span>
                    </div>
                    <div className="header-cell conf-cell">
                        <label>Model Confidence</label>
                        <ConfidenceBar value={disease_confidence || 0} />
                    </div>
                    <div className="header-cell">
                        <label>Processed In</label>
                        <span className="proc-time">{processing_time_seconds ?? '—'}s</span>
                    </div>
                </div>
            </div>

            {/* Patient Explanation */}
            <div className="section-card">
                <h2>💬 Patient Explanation</h2>
                <p className="explanation-text">{explanation?.simple_explanation}</p>
                {explanation?.physiological_process && (
                    <div className="sub-section">
                        <h3>How it affects your body</h3>
                        <p className="explanation-sub">{explanation.physiological_process}</p>
                    </div>
                )}
                {explanation?.plain_language_takeaway && (
                    <div className="takeaway-box">
                        <span className="takeaway-icon">💡</span>
                        <p>{explanation.plain_language_takeaway}</p>
                    </div>
                )}
            </div>

            {/* Risk Details */}
            <div className="section-card">
                <h2>⚠️ Risk Assessment</h2>
                <p className="scope-note">{risk?.scope_note}</p>
                <div className="risk-reasons">
                    {(risk?.reasons || []).map((r, i) => (
                        <div key={i} className={`reason-row reason-${riskColor}`}>
                            <span className="reason-num">{i + 1}</span>
                            <span>{r}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Recommendations */}
            <div className="result-grid">
                <div className="section-card">
                    <h2>✅ Safe Next Steps</h2>
                    <TagList items={recommendation?.safe_next_steps} variant="success" />
                </div>
                <div className="section-card">
                    <h2>🚨 Urgent Warning Signs</h2>
                    <TagList items={recommendation?.urgent_attention_signs} variant="danger" />
                </div>
            </div>

            {/* Justification */}
            <div className="section-card">
                <h2>📝 Clinical Justification</h2>
                <div className="just-conf">
                    <span>AI Confidence</span>
                    <ConfidenceBar value={justification?.confidence || 0} />
                </div>
                <div className="just-grid">
                    <div>
                        <h3>Rationale</h3>
                        <ul className="just-list">
                            {(justification?.rationale || []).map((r, i) => <li key={i}>{r}</li>)}
                        </ul>
                    </div>
                    <div>
                        <h3>Limitations</h3>
                        <ul className="just-list just-limitations">
                            {(justification?.limitations || []).map((l, i) => <li key={i}>{l}</li>)}
                        </ul>
                    </div>
                </div>
            </div>

            {/* Verification */}
            <div className="section-card verification-card">
                <h2>🔍 Verification</h2>
                <div className="verif-score-row">
                    <span>Consistency Score</span>
                    <ConfidenceBar value={verification?.consistency_score || 0} />
                </div>
                {verification?.contradictions?.length > 0 && (
                    <div className="contradictions">
                        <h3>Contradictions detected</h3>
                        {verification.contradictions.map((c, i) => (
                            <div key={i} className="contradiction-item">{c}</div>
                        ))}
                    </div>
                )}
                {verification?.safety_notes?.length > 0 && (
                    <div className="safety-notes">
                        {verification.safety_notes.map((n, i) => (
                            <div key={i} className="safety-note">{n}</div>
                        ))}
                    </div>
                )}
            </div>

            {/* Disclaimer */}
            <div className="disclaimer">
                <span>⚕️</span>
                <p>This system provides <strong>informational support only</strong> and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek guidance from a qualified healthcare provider.</p>
            </div>
        </div>
    );
}
