import React from 'react';
import './ArchitecturePage.css';

const PIPELINE = [
    {
        icon: '📄',
        title: 'Medical Report Input',
        desc: 'User pastes a clinical discharge summary or uploads a .txt file. The raw text is sent to the FastAPI backend.',
        color: '#38bdf8',
        tag: 'Input',
    },
    {
        icon: '🧬',
        title: 'BioClinicalBERT — Clinical Extraction',
        desc: 'Extracts structured clinical facts: diagnoses, abnormal findings, key observations. Predicts the disease category with a confidence score using a fine-tuned classifier head.',
        color: '#818cf8',
        tag: 'Classification',
        detail: 'Model: emilyalsentzer/Bio_ClinicalBERT',
    },
    {
        icon: '⚠️',
        title: 'Risk Agent',
        desc: 'Analyzes clinical findings to assign Low / Medium / High risk. Produces 3 specific clinical observations as reasons.',
        color: '#f87171',
        tag: 'FLAN-T5',
        detail: 'Heuristic fallback with clinical keyword scoring',
    },
    {
        icon: '💬',
        title: 'Explainability Agent',
        desc: 'Converts clinical facts into a patient-friendly explanation at a 6th-8th grade reading level. Covers: disease, body impact, risks, and follow-up importance.',
        color: '#34d399',
        tag: 'FLAN-T5',
        detail: '3-attempt generation + heuristic fallback',
    },
    {
        icon: '✅',
        title: 'Recommendation Agent',
        desc: 'Generates safe, non-prescriptive next steps. Strict rules: no diagnosis, no drugs, no dosage. Enforces 3-5 safety steps and 2 urgent warning signs.',
        color: '#38bdf8',
        tag: 'FLAN-T5',
        detail: 'Safety-first prompt engineering',
    },
    {
        icon: '📝',
        title: 'Justification Agent',
        desc: 'Explains WHY each recommendation was made. Returns structured rationale list, confidence score, and known limitations.',
        color: '#fbbf24',
        tag: 'FLAN-T5',
        detail: 'Transparency for clinical reviewers',
    },
    {
        icon: '🔍',
        title: 'Verification Agent',
        desc: 'Cross-checks all 5 agent outputs for contradictions. Computes a consistency score (0-1). Falls back to heuristic checking if LLM is inconclusive.',
        color: '#818cf8',
        tag: 'FLAN-T5',
        detail: 'Cross-agent consistency validation',
    },
    {
        icon: '🛡️',
        title: 'Output Validator + Schema',
        desc: 'Validates the final JSON against the full schema. Checks field lengths, confidence range, risk_level enum. If validation fails → regenerates once.',
        color: '#34d399',
        tag: 'Python',
        detail: 'validate_final_output() + build_final_output()',
    },
    {
        icon: '📊',
        title: 'Structured Response',
        desc: 'Returns a guaranteed-complete JSON with all 7 sections: disease_category, risk, explanation, recommendation, justification, verification.',
        color: '#38bdf8',
        tag: 'Output',
    },
];

export default function ArchitecturePage() {
    return (
        <main className="page-container">
            <div className="page-header">
                <h1>System Architecture</h1>
                <p>Six-agent clinical NLP pipeline powering real-time medical report analysis</p>
            </div>

            {/* Tech stack badges */}
            <div className="tech-stack section-card">
                <h2>⚙️ Technology Stack</h2>
                <div className="tech-grid">
                    {[
                        { name: 'BioClinicalBERT', note: 'Clinical extraction & classification', color: '#818cf8' },
                        { name: 'FLAN-T5', note: 'Multi-agent reasoning (5 agents)', color: '#38bdf8' },
                        { name: 'FastAPI', note: 'REST API backend + CORS', color: '#34d399' },
                        { name: 'React 18', note: 'Real-time frontend UI', color: '#fbbf24' },
                        { name: 'PyTorch', note: 'Model inference', color: '#f87171' },
                        { name: 'MIMIC-III', note: 'Clinical training dataset', color: '#94a3b8' },
                    ].map(t => (
                        <div key={t.name} className="tech-card" style={{ borderColor: `${t.color}30` }}>
                            <span className="tech-name" style={{ color: t.color }}>{t.name}</span>
                            <span className="tech-note">{t.note}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Pipeline */}
            <div className="pipeline-section">
                <h2 className="pipeline-heading">Pipeline Flow</h2>
                {PIPELINE.map((step, i) => (
                    <div key={i} className="pipeline-step fade-in" style={{ animationDelay: `${i * 0.05}s` }}>
                        <div className="step-connector">
                            <div className="step-icon-wrap" style={{ background: `${step.color}20`, borderColor: `${step.color}50` }}>
                                <span className="step-icon">{step.icon}</span>
                            </div>
                            {i < PIPELINE.length - 1 && <div className="connector-line" />}
                        </div>
                        <div className="step-body">
                            <div className="step-header">
                                <span className="step-badge" style={{ background: `${step.color}20`, color: step.color }}>{step.tag}</span>
                                <h3 className="step-title">{step.title}</h3>
                            </div>
                            <p className="step-desc">{step.desc}</p>
                            {step.detail && <span className="step-detail">{step.detail}</span>}
                        </div>
                    </div>
                ))}
            </div>
        </main>
    );
}
