import React, { useState, useRef } from 'react';
import './ReportInput.css';

const SAMPLE_REPORT = `Discharge Summary - Patient: [**Known lastname 742**]
Admission Date: [**2198-8-7**]   Discharge Date: [**2198-8-17**]
Date of Birth:  [**2131-4-17**]   Sex: M   Service: Medicine

Chief Complaint: Sepsis, Acute Kidney Injury

History of Present Illness: 67-year-old male with a history of type 2 diabetes mellitus and chronic kidney disease (stage 3) who presented to the emergency department with 3 days of fever (T 39.2°C), rigors, and altered mental status. Blood cultures obtained on admission returned positive for gram-negative bacteremia consistent with E. coli. He was admitted to the ICU for clinical monitoring.

Pertinent Results: 
WBC 18.2, Hgb 10.1, Plt 84, Creatinine 3.8 (baseline 1.6), Lactate 4.1, Blood culture: E. coli sensitive to cefepime.

Brief Clinical Course:
Patient was started on broad-spectrum antibiotics and fluid resuscitation. Creatinine improved from 3.8 to 2.1 over the course of admission. Patient was step-down from ICU on day 5. He remained afebrile for 48 hours prior to discharge.

Discharge Diagnoses:
1. Sepsis secondary to gram-negative bacteremia (E. coli)
2. Acute kidney injury, recovering
3. Type 2 diabetes mellitus, uncontrolled
4. Thrombocytopenia, likely reactive

Discharge Instructions: Follow up with primary care in 1 week. Labs in 3 days.`;

export default function ReportInput({ onSubmit, loading }) {
    const [text, setText] = useState('');
    const [charCount, setCharCount] = useState(0);
    const fileRef = useRef(null);

    const handleTextChange = (e) => {
        setText(e.target.value);
        setCharCount(e.target.value.length);
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const content = ev.target.result;
            setText(content);
            setCharCount(content.length);
        };
        reader.readAsText(file);
    };

    const loadSample = () => {
        setText(SAMPLE_REPORT);
        setCharCount(SAMPLE_REPORT.length);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (text.trim().length < 10) return;
        onSubmit(text.trim());
    };

    return (
        <form className="report-input-card section-card" onSubmit={handleSubmit}>
            <div className="input-header">
                <h2>📋 Medical Report Input</h2>
                <div className="input-actions">
                    <button type="button" className="btn-secondary" onClick={loadSample}>
                        📄 Load Sample
                    </button>
                    <button type="button" className="btn-secondary" onClick={() => fileRef.current?.click()}>
                        📁 Upload .txt
                    </button>
                    <input
                        ref={fileRef}
                        type="file"
                        accept=".txt,.text"
                        style={{ display: 'none' }}
                        onChange={handleFileUpload}
                    />
                </div>
            </div>

            <textarea
                className="report-textarea"
                placeholder="Paste a clinical discharge summary or medical report here…&#10;&#10;(Supports MIMIC-III style discharge notes, free-text clinical notes, or any structured clinical text)"
                value={text}
                onChange={handleTextChange}
                rows={14}
                disabled={loading}
                required
            />

            <div className="input-footer">
                <span className="char-count">{charCount.toLocaleString()} characters</span>
                {text.length > 0 && (
                    <button
                        type="button"
                        className="btn-secondary btn-clear"
                        onClick={() => { setText(''); setCharCount(0); }}
                    >
                        ✕ Clear
                    </button>
                )}
                <button
                    type="submit"
                    className="btn-primary analyze-btn"
                    disabled={loading || text.trim().length < 10}
                >
                    {loading ? (
                        <>
                            <span className="btn-spinner" />
                            Analyzing…
                        </>
                    ) : (
                        <>🔬 Analyze Report</>
                    )}
                </button>
            </div>
        </form>
    );
}
