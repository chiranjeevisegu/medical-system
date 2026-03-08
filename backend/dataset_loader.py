from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import pandas as pd


@dataclass
class MIMICDatasetLoader:
    dataset_dir: Path
    output_path: Path
    max_cases: int = 20

    def prepare_samples(self, force_refresh: bool = False) -> list[dict]:
        if self.output_path.exists() and not force_refresh:
            with self.output_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                samples = data[: self.max_cases]
                has_context = all(
                    "diagnosis_context" in sample and "lab_context" in sample
                    for sample in samples
                )
                has_temporal = all("previous_report_text" in sample for sample in samples)
                if not has_context:
                    self._attach_optional_contexts(samples)
                if not has_temporal:
                    self._attach_previous_reports(samples)
                if not has_context or not has_temporal:
                    with self.output_path.open("w", encoding="utf-8") as f:
                        json.dump(samples, f, ensure_ascii=False, indent=2)
                return samples

        samples = self._extract_discharge_summaries()
        if not samples:
            samples = self._fallback_build_from_admissions()
        if not samples:
            raise RuntimeError("No usable MIMIC demo records found in dataset folder.")
        self._attach_optional_contexts(samples)
        self._attach_previous_reports(samples)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(samples[: self.max_cases], f, ensure_ascii=False, indent=2)

        return samples[: self.max_cases]

    def _extract_discharge_summaries(self) -> list[dict]:
        noteevents_path = self.dataset_dir / "NOTEEVENTS.csv"
        if not noteevents_path.exists():
            return []

        extracted: list[dict] = []
        chunks = pd.read_csv(noteevents_path, chunksize=5000, low_memory=False)

        for chunk in chunks:
            cols = {c.lower(): c for c in chunk.columns}
            if "category" not in cols or "text" not in cols:
                continue

            category_col = cols["category"]
            text_col = cols["text"]
            hadm_col = cols.get("hadm_id")
            subject_col = cols.get("subject_id")
            chartdate_col = cols.get("chartdate")

            filtered = chunk[
                chunk[category_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("discharge summary")
            ]
            for idx, row in filtered.iterrows():
                text = self._clean_text(row.get(text_col, ""))
                if not text:
                    continue

                hadm = row.get(hadm_col) if hadm_col else ""
                case_id = f"case_{len(extracted) + 1:03d}_{str(hadm) if pd.notna(hadm) else idx}"
                extracted.append(
                    {
                        "case_id": case_id,
                        "subject_id": self._safe_value(row.get(subject_col) if subject_col else None),
                        "hadm_id": self._safe_value(hadm),
                        "chartdate": self._safe_value(row.get(chartdate_col) if chartdate_col else None),
                        "report_text": text,
                        "source": "NOTEEVENTS_discharge_summary",
                    }
                )
                if len(extracted) >= self.max_cases:
                    return extracted

        return extracted

    def _fallback_build_from_admissions(self) -> list[dict]:
        admissions_path = self.dataset_dir / "ADMISSIONS.csv"
        diagnoses_path = self.dataset_dir / "DIAGNOSES_ICD.csv"
        d_icd_path = self.dataset_dir / "D_ICD_DIAGNOSES.csv"

        if not admissions_path.exists():
            return []

        admissions = pd.read_csv(admissions_path, low_memory=False)
        diagnosis_map: dict[str, list[str]] = {}

        if diagnoses_path.exists():
            diag = pd.read_csv(diagnoses_path, low_memory=False)
            diag_cols = {c.lower(): c for c in diag.columns}
            hadm_col = diag_cols.get("hadm_id")
            icd_col = diag_cols.get("icd9_code")

            long_title_map: dict[str, str] = {}
            if d_icd_path.exists() and icd_col:
                d_icd = pd.read_csv(d_icd_path, low_memory=False)
                d_cols = {c.lower(): c for c in d_icd.columns}
                d_icd_col = d_cols.get("icd9_code")
                long_title_col = d_cols.get("long_title")
                if d_icd_col and long_title_col:
                    long_title_map = {
                        str(r[d_icd_col]).strip(): str(r[long_title_col]).strip()
                        for _, r in d_icd.iterrows()
                        if pd.notna(r[d_icd_col]) and pd.notna(r[long_title_col])
                    }

            if hadm_col and icd_col:
                for _, row in diag.iterrows():
                    hadm = str(row[hadm_col]).strip()
                    icd = str(row[icd_col]).strip()
                    if not hadm or not icd:
                        continue
                    text = long_title_map.get(icd, f"ICD9 {icd}")
                    diagnosis_map.setdefault(hadm, []).append(text)

        cols = {c.lower(): c for c in admissions.columns}
        hadm_col = cols.get("hadm_id")
        subj_col = cols.get("subject_id")
        admit_col = cols.get("admittime")
        disch_col = cols.get("dischtime")
        diagnosis_col = cols.get("diagnosis")

        if not hadm_col:
            return []

        samples: list[dict] = []
        for _, row in admissions.head(self.max_cases * 3).iterrows():
            hadm = str(row[hadm_col]).strip()
            if not hadm:
                continue

            primary = str(row.get(diagnosis_col, "")).strip() if diagnosis_col else ""
            icd_terms = diagnosis_map.get(hadm, [])[:8]
            report_text = self._clean_text(
                " ".join(
                    [
                        f"Admission summary for hospital admission {hadm}.",
                        f"Primary diagnosis: {primary}." if primary else "",
                        f"Associated ICD diagnoses: {', '.join(icd_terms)}." if icd_terms else "",
                        f"Admission time: {row.get(admit_col)}." if admit_col else "",
                        f"Discharge time: {row.get(disch_col)}." if disch_col else "",
                    ]
                )
            )
            if not report_text:
                continue

            samples.append(
                {
                    "case_id": f"case_{len(samples) + 1:03d}_{hadm}",
                    "subject_id": self._safe_value(row.get(subj_col) if subj_col else None),
                    "hadm_id": hadm,
                    "chartdate": self._safe_value(row.get(disch_col) if disch_col else None),
                    "report_text": report_text,
                    "source": "ADMISSIONS_DIAGNOSES_fallback",
                }
            )
            if len(samples) >= self.max_cases:
                break

        return samples

    def _attach_optional_contexts(self, samples: list[dict]) -> None:
        hadm_ids = {
            str(sample.get("hadm_id")).strip()
            for sample in samples
            if sample.get("hadm_id")
        }
        if not hadm_ids:
            for sample in samples:
                sample["diagnosis_context"] = []
                sample["lab_context"] = []
                sample["lab_context_text"] = ""
                sample["medications_context"] = []
                sample["medications_context_text"] = ""
            return

        diagnosis_map = self._load_diagnosis_context_map(hadm_ids)
        lab_map = self._load_lab_context_map(hadm_ids)
        prescriptions_map = self._load_prescriptions_map(hadm_ids)

        for sample in samples:
            hadm = str(sample.get("hadm_id", "")).strip()
            diagnoses = diagnosis_map.get(hadm, [])
            labs = lab_map.get(hadm, [])
            meds = prescriptions_map.get(hadm, [])
            sample["diagnosis_context"] = diagnoses
            sample["lab_context"] = labs
            sample["lab_context_text"] = self._build_lab_context_text(labs)
            sample["medications_context"] = meds
            sample["medications_context_text"] = self._build_medications_context_text(meds)

    def _attach_previous_reports(self, samples: list[dict]) -> None:
        by_subject: dict[str, list[dict]] = {}
        for sample in samples:
            subject_id = str(sample.get("subject_id") or "").strip()
            if not subject_id:
                sample["previous_report_text"] = ""
                continue
            by_subject.setdefault(subject_id, []).append(sample)

        for subject_samples in by_subject.values():
            subject_samples.sort(
                key=lambda x: (
                    str(x.get("chartdate") or ""),
                    str(x.get("hadm_id") or ""),
                )
            )
            previous_text = ""
            for sample in subject_samples:
                sample["previous_report_text"] = previous_text
                current_text = self._clean_text(sample.get("report_text", ""))
                if current_text:
                    previous_text = current_text

        for sample in samples:
            sample.setdefault("previous_report_text", "")

    def _load_diagnosis_context_map(self, hadm_ids: set[str]) -> dict[str, list[str]]:
        diagnoses_path = self.dataset_dir / "DIAGNOSES_ICD.csv"
        dictionary_path = self.dataset_dir / "D_ICD_DIAGNOSES.csv"
        if not diagnoses_path.exists():
            return {}

        icd_to_title: dict[str, str] = {}
        if dictionary_path.exists():
            d_icd = pd.read_csv(dictionary_path, low_memory=False)
            d_cols = {c.lower(): c for c in d_icd.columns}
            code_col = d_cols.get("icd9_code")
            title_col = d_cols.get("long_title")
            if code_col and title_col:
                for _, row in d_icd.iterrows():
                    code = self._normalize_icd_code(row.get(code_col))
                    title = self._clean_text(row.get(title_col))
                    if code and title:
                        icd_to_title[code] = title

        diagnosis_map: dict[str, list[str]] = {hadm: [] for hadm in hadm_ids}
        diag = pd.read_csv(diagnoses_path, low_memory=False)
        cols = {c.lower(): c for c in diag.columns}
        hadm_col = cols.get("hadm_id")
        icd_col = cols.get("icd9_code")
        seq_col = cols.get("seq_num")
        if not hadm_col or not icd_col:
            return diagnosis_map

        filtered = diag[diag[hadm_col].astype(str).isin(hadm_ids)].copy()
        if seq_col:
            filtered = filtered.sort_values(by=[hadm_col, seq_col], na_position="last")

        for _, row in filtered.iterrows():
            hadm = str(row.get(hadm_col, "")).strip()
            code = self._normalize_icd_code(row.get(icd_col))
            if not hadm or not code:
                continue
            term = icd_to_title.get(code, f"ICD9 {code}")
            if term not in diagnosis_map[hadm]:
                diagnosis_map[hadm].append(term)

        for hadm, values in diagnosis_map.items():
            diagnosis_map[hadm] = values[:15]
        return diagnosis_map

    def _load_lab_context_map(self, hadm_ids: set[str]) -> dict[str, list[dict]]:
        labevents_path = self.dataset_dir / "LABEVENTS.csv"
        d_labitems_path = self.dataset_dir / "D_LABITEMS.csv"
        if not labevents_path.exists():
            return {}

        item_name_map: dict[str, str] = {}
        if d_labitems_path.exists():
            d_lab = pd.read_csv(d_labitems_path, low_memory=False)
            d_cols = {c.lower(): c for c in d_lab.columns}
            itemid_col = d_cols.get("itemid")
            label_col = d_cols.get("label")
            if itemid_col and label_col:
                for _, row in d_lab.iterrows():
                    itemid = self._safe_value(row.get(itemid_col))
                    label = self._clean_text(row.get(label_col))
                    if itemid and label:
                        item_name_map[itemid] = label

        lab_map: dict[str, list[dict]] = {hadm: [] for hadm in hadm_ids}
        chunk_iter = pd.read_csv(labevents_path, chunksize=25000, low_memory=False)

        for chunk in chunk_iter:
            cols = {c.lower(): c for c in chunk.columns}
            hadm_col = cols.get("hadm_id")
            itemid_col = cols.get("itemid")
            value_col = cols.get("value")
            valuenum_col = cols.get("valuenum")
            uom_col = cols.get("valueuom")
            flag_col = cols.get("flag")
            charttime_col = cols.get("charttime")
            if not hadm_col or not itemid_col:
                continue

            chunk_filtered = chunk[chunk[hadm_col].astype(str).isin(hadm_ids)]
            for _, row in chunk_filtered.iterrows():
                hadm = str(row.get(hadm_col, "")).strip()
                if not hadm:
                    continue
                if len(lab_map[hadm]) >= 25:
                    continue

                itemid = self._safe_value(row.get(itemid_col))
                item_label = item_name_map.get(itemid or "", f"Lab item {itemid}" if itemid else "Unknown lab")
                value = self._clean_text(row.get(value_col))
                value_num = self._safe_value(row.get(valuenum_col))
                value_uom = self._clean_text(row.get(uom_col))
                flag = self._clean_text(row.get(flag_col))
                charttime = self._safe_value(row.get(charttime_col))

                lab_map[hadm].append(
                    {
                        "item": item_label,
                        "value": value_num or value,
                        "unit": value_uom,
                        "flag": flag,
                        "charttime": charttime,
                    }
                )

        return lab_map

    @staticmethod
    def _build_lab_context_text(labs: list[dict]) -> str:
        if not labs:
            return ""
        lines: list[str] = []
        for lab in labs[:15]:
            item = str(lab.get("item", "")).strip()
            value = str(lab.get("value", "")).strip()
            unit = str(lab.get("unit", "")).strip()
            flag = str(lab.get("flag", "")).strip()
            time = str(lab.get("charttime", "")).strip()
            parts = [item]
            if value:
                parts.append(f"value {value}{(' ' + unit) if unit else ''}")
            if flag:
                parts.append(f"flag {flag}")
            if time:
                parts.append(f"time {time}")
            lines.append(", ".join(parts))
        return " | ".join(lines)

    def _load_prescriptions_map(self, hadm_ids: set[str]) -> dict[str, list[dict]]:
        """Read PRESCRIPTIONS.csv and return a map of hadm_id -> list of medication dicts."""
        prescriptions_path = self.dataset_dir / "PRESCRIPTIONS.csv"
        if not prescriptions_path.exists():
            return {hadm: [] for hadm in hadm_ids}

        med_map: dict[str, list[dict]] = {hadm: [] for hadm in hadm_ids}
        try:
            chunk_iter = pd.read_csv(prescriptions_path, chunksize=25000, low_memory=False)
            for chunk in chunk_iter:
                cols = {c.lower(): c for c in chunk.columns}
                hadm_col = cols.get("hadm_id")
                drug_col = cols.get("drug")
                route_col = cols.get("route")
                start_col = cols.get("startdate") or cols.get("starttime")
                if not hadm_col or not drug_col:
                    continue

                chunk_filtered = chunk[chunk[hadm_col].astype(str).isin(hadm_ids)]
                for _, row in chunk_filtered.iterrows():
                    hadm = str(row.get(hadm_col, "")).strip()
                    if not hadm:
                        continue
                    if len(med_map[hadm]) >= 10:
                        continue
                    drug = self._clean_text(row.get(drug_col))
                    route = self._clean_text(row.get(route_col)) if route_col else ""
                    start = self._safe_value(row.get(start_col)) if start_col else None
                    if drug:
                        med_map[hadm].append({"drug": drug, "route": route, "start": start})
        except Exception:  # noqa: BLE001
            pass  # PRESCRIPTIONS.csv missing or malformed – return empty maps
        return med_map

    @staticmethod
    def _build_medications_context_text(meds: list[dict]) -> str:
        """Format medications list as a compact pipe-separated string."""
        if not meds:
            return ""
        parts: list[str] = []
        for med in meds[:10]:
            drug = str(med.get("drug", "")).strip()
            route = str(med.get("route", "")).strip()
            if drug:
                parts.append(f"{drug} ({route})" if route else drug)
        return " | ".join(parts)

    @staticmethod
    def _normalize_icd_code(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        code = str(value).strip()
        if not code:
            return ""
        if code.endswith(".0"):
            code = code[:-2]
        return code

    @staticmethod
    def _clean_text(text: object) -> str:
        if text is None:
            return ""
        s = str(text)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _safe_value(value: object) -> str | None:
        if value is None:
            return None
        s = str(value).strip()
        if s == "" or s.lower() == "nan":
            return None
        return s
