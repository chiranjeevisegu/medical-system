import React from 'react';
import './AboutPage.css';

const TEAM_FEATURES = [
    { icon: '🧬', title: 'Clinical NLP', desc: 'BioClinicalBERT fine-tuned on MIMIC-III discharge summaries for structured clinical fact extraction and 7-class disease classification.' },
    { icon: '🤖', title: 'Multi-Agent Reasoning', desc: 'FLAN-T5 powers five specialized agents: Risk, Explainability, Recommendation, Justification, and Verification — each with strict safety constraints.' },
    { icon: '🛡️', title: 'Safety-First Design', desc: 'Every recommendation is filtered through no-diagnose / no-prescribe rules. A dedicated verification agent checks cross-agent consistency.' },
    { icon: '📊', title: 'Semantic Evaluation', desc: 'Beyond BLEU/ROUGE, the system measures cosine similarity with BioClinicalBERT embeddings (semantic QA score) and FKGL readability improvement.' },
];

const DATASET_INFO = [
    { label: 'Name', value: 'MIMIC-III Clinical Database Demo v1.4' },
    { label: 'Publisher', value: 'MIT Lab for Computational Physiology' },
    { label: 'Data Type', value: 'De-identified ICU clinical notes (NOTEEVENTS)' },
    { label: 'Additional', value: 'DIAGNOSES_ICD, LABEVENTS, PRESCRIPTIONS tables' },
];

const MODELS = [
    {
        name: 'Bio_ClinicalBERT',
        hf: 'emilyalsentzer/Bio_ClinicalBERT',
        role: 'Clinical feature extraction + disease classification',
        color: '#818cf8',
        detail: 'BERT pre-trained on MIMIC-III and PubMed notes. Fine-tuned with a linear classifier head for 7 disease categories.',
    },
    {
        name: 'FLAN-T5',
        hf: 'google/flan-t5-base',
        role: 'Multi-agent reasoning (5 agents)',
        color: '#38bdf8',
        detail: 'Instruction-following text-to-text transformer. Used for all reasoning agents with strict JSON-only prompt engineering.',
    },
];

export default function AboutPage() {
    return (
        <main className="page-container">
            <div className="page-header">
                <h1>About This System</h1>
                <p>Explainable multi-agent medical report understanding for patient education and clinical transparency</p>
            </div>

            {/* Project description */}
            <div className="section-card about-hero-card">
                <div className="about-hero-inner">
                    <div className="about-text">
                        <h2>🧠 Project Overview</h2>
                        <p>
                            This system addresses a critical gap in healthcare: patients often receive dense,
                            jargon-heavy discharge summaries that they cannot understand. By combining
                            <strong> BioClinicalBERT</strong> and <strong>FLAN-T5</strong> in a structured
                            multi-agent pipeline, we automatically convert complex clinical notes into
                            patient-friendly explanations — with risk assessment, safe recommendations,
                            and full clinical justification.
                        </p>
                        <p style={{ marginTop: 12 }}>
                            The system is trained and evaluated on the <strong>MIMIC-III clinical database</strong>,
                            a gold-standard de-identified ICU dataset widely used in clinical NLP research.
                        </p>
                    </div>
                    <div className="about-badge-stack">
                        <div className="about-badge">Clinical NLP</div>
                        <div className="about-badge">Multi-Agent AI</div>
                        <div className="about-badge">MIMIC-III</div>
                        <div className="about-badge">Patient Safety</div>
                        <div className="about-badge">Explainable AI</div>
                    </div>
                </div>
            </div>

            {/* Feature highlights */}
            <div className="features-grid">
                {TEAM_FEATURES.map((f, i) => (
                    <div key={i} className="feature-card section-card">
                        <span className="feature-icon">{f.icon}</span>
                        <h3>{f.title}</h3>
                        <p>{f.desc}</p>
                    </div>
                ))}
            </div>

            {/* Models */}
            <section className="section-card models-section">
                <h2>🔬 Models Used</h2>
                <div className="models-grid">
                    {MODELS.map((m, i) => (
                        <div key={i} className="model-card" style={{ borderColor: `${m.color}30` }}>
                            <div className="model-header">
                                <span className="model-name" style={{ color: m.color }}>{m.name}</span>
                                <a href={`https://huggingface.co/${m.hf}`} target="_blank" rel="noreferrer" className="model-hf">
                                    🤗 HuggingFace ↗
                                </a>
                            </div>
                            <code className="model-id">{m.hf}</code>
                            <p className="model-role"><strong>{m.role}</strong></p>
                            <p className="model-detail">{m.detail}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Dataset */}
            <section className="section-card dataset-section">
                <h2>📁 Dataset — MIMIC-III</h2>
                <div className="dataset-table">
                    {DATASET_INFO.map((row, i) => (
                        <div key={i} className="dataset-row">
                            <span className="ds-label">{row.label}</span>
                            <span className="ds-value">{row.value}</span>
                        </div>
                    ))}
                </div>
            </section>

            {/* Goal */}
            <div className="goal-card section-card">
                <span className="goal-icon">🎯</span>
                <div>
                    <h3>Research Goal</h3>
                    <p>Make complex medical reports <strong>understandable for patients</strong> while maintaining clinical accuracy, safety constraints, and transparency through multi-agent explainability.</p>
                </div>
            </div>

            {/* Pipeline reminder */}
            <div className="pipeline-mini section-card">
                <h2>↔️ Quick Reference — Pipeline in 6 Steps</h2>
                <div className="pipeline-mini-grid">
                    {['Clinical Extraction', 'Risk Assessment', 'Plain-Language Explanation',
                        'Safe Recommendations', 'Clinical Justification', 'Consistency Verification'
                    ].map((s, i) => (
                        <div key={i} className="mini-step">
                            <span className="mini-num">{i + 1}</span>
                            <span className="mini-label">{s}</span>
                        </div>
                    ))}
                </div>
            </div>
        </main>
    );
}
