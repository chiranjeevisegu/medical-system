import React, { useState, useEffect } from 'react';
import './HistoryPanel.css';

const STORAGE_KEY = 'medai_history';
const MAX_HISTORY = 20;

export function saveToHistory(reportText, result) {
    try {
        const history = getHistory();
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            preview: reportText.slice(0, 160).replace(/\s+/g, ' ').trim() + '…',
            disease_category: result.disease_category || 'Unknown',
            disease_confidence: result.disease_confidence || 0,
            risk_level: result.risk?.risk_level || 'Unknown',
            result,
        };
        const updated = [entry, ...history].slice(0, MAX_HISTORY);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        return updated;
    } catch { return []; }
}

export function getHistory() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch { return []; }
}

export function clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
}

const RISK_COLORS = { High: '#f87171', Medium: '#fbbf24', Low: '#34d399' };

function timeAgo(iso) {
    const diff = Date.now() - new Date(iso).getTime();
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
}

export default function HistoryPanel({ onSelect, refreshKey }) {
    const [history, setHistory] = useState([]);

    const refresh = () => setHistory(getHistory());

    useEffect(() => {
        refresh();
        window.addEventListener('storage', refresh);
        window.addEventListener('focus', refresh);
        return () => {
            window.removeEventListener('storage', refresh);
            window.removeEventListener('focus', refresh);
        };
    }, []);

    // Refresh whenever parent saves a new analysis
    useEffect(() => { refresh(); }, [refreshKey]);

    return (
        <div className="history-section">
            <div className="history-section-header">
                <div className="history-title-row">
                    <h2>📂 Previous Analyses</h2>
                    <div className="history-meta">
                        {history.length > 0 && (
                            <>
                                <span className="history-count-badge">
                                    {history.length} report{history.length !== 1 ? 's' : ''}
                                </span>
                                <button
                                    className="history-clear-btn"
                                    onClick={() => { clearHistory(); setHistory([]); }}
                                >
                                    Clear All
                                </button>
                            </>
                        )}
                    </div>
                </div>
                <p className="history-subtitle">
                    {history.length === 0
                        ? 'Analyses you run will be saved here automatically — click any card to restore results.'
                        : 'Click any card below to instantly restore its full analysis results.'}
                </p>
            </div>

            {/* Empty state */}
            {history.length === 0 && (
                <div className="history-empty">
                    <span className="history-empty-icon">🔬</span>
                    <p>No reports analysed yet. Paste a clinical discharge summary above and click <strong>Analyse Report</strong>.</p>
                </div>
            )}

            {/* History grid */}
            {history.length > 0 && (
                <div className="history-grid">
                    {history.map((entry) => (
                        <div
                            key={entry.id}
                            className="history-card"
                            onClick={() => onSelect(entry.result)}
                            role="button"
                            tabIndex={0}
                            onKeyDown={e => e.key === 'Enter' && onSelect(entry.result)}
                        >
                            <div className="hcard-top">
                                <span className="hcard-risk" style={{ color: RISK_COLORS[entry.risk_level] || '#94a3b8' }}>
                                    ● {entry.risk_level} Risk
                                </span>
                                <span className="hcard-time">{timeAgo(entry.timestamp)}</span>
                            </div>

                            <div className="hcard-category">
                                <span className="badge badge-info">{entry.disease_category}</span>
                                <span className="hcard-conf">{Math.round(entry.disease_confidence * 100)}% confidence</span>
                            </div>

                            <p className="hcard-preview">{entry.preview}</p>

                            <div className="hcard-cta">View Analysis →</div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
