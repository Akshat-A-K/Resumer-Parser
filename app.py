"""T13.1 Smart Document Parser - Streamlit app."""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluator import annotations_to_entity, evaluate_batch, load_ground_truth, results_to_dataframe
from llm_parser import extract_entities
from ocr_extractor import (
    extract_text_and_links_from_bytes,
    extract_text_from_image_bytes,
    extract_urls_from_text,
)
from schema import FIELD_SPECS, LIST_FIELDS, ResumeEntity, default_selected_fields

load_dotenv()

st.set_page_config(page_title="T13.1 Resume Parser", page_icon="📄", layout="wide")
st.title("T13.1 - Smart Document Parser")


def parse_uploaded_file(uploaded_file) -> tuple[str, list[str]]:
    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.read()

    if suffix == ".pdf":
        return extract_text_and_links_from_bytes(payload)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        text = extract_text_from_image_bytes(payload)
        links = extract_urls_from_text(text) if text else []
        return text, links
    return "[ERROR: Unsupported file format]", []


def entity_to_row(entity: ResumeEntity, selected_fields: list[str]) -> dict:
    data = entity.model_dump()
    row = {}
    for field in selected_fields:
        value = data.get(field)
        if isinstance(value, list):
            row[field] = ", ".join(value)
        else:
            row[field] = value
    return row


def parse_list_input(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _selected_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _find_ollama_cli() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@st.cache_data(ttl=60)
def get_ollama_model_options(current_model: str) -> list[str]:
    options: list[str] = ["llama3.2:3b", "resume-parser-local", current_model]
    ollama_cli = _find_ollama_cli()

    if ollama_cli:
        try:
            result = subprocess.run([ollama_cli, "list"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                for line in lines[1:]:
                    model_name = line.split()[0]
                    options.append(model_name)
        except Exception:
            pass

    deduped = []
    for item in options:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def model_option_label(model_name: str, provider_name: str) -> str:
    name = (model_name or "").strip()
    low = name.lower()

    if provider_name == "ollama-local":
        if "resume-parser" in low:
            return f"{name}  (Recommended)"

    return name


with st.sidebar:
    st.header("Workflow")
    mode = st.radio("Choose task", ["Parse One Resume", "Evaluate Quality"], index=0)

    st.markdown("---")
    st.header("Model Provider")
    provider = "ollama-local"
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    ollama_model_options = get_ollama_model_options(ollama_model)

    ollama_model = st.selectbox(
        "OLLAMA_MODEL",
        options=ollama_model_options,
        index=_selected_index(ollama_model_options, ollama_model),
        format_func=lambda m: model_option_label(m, "ollama-local"),
    )

if mode == "Parse One Resume":
    st.subheader("Parse One Resume")
    st.caption("Step 1: choose fields. Step 2: upload resume. Step 3: run extraction and edit results.")

    all_fields = list(FIELD_SPECS.keys())
    selected_fields = st.multiselect(
        "Select fields to extract (15-20 options available)",
        options=all_fields,
        default=default_selected_fields(),
        format_func=lambda x: FIELD_SPECS[x]["label"],
    )

    if not selected_fields:
        st.warning("Select at least one field.")
        st.stop()

    uploaded = st.file_uploader("Step 2: Upload resume", type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"])

    if uploaded is not None:
        with st.spinner("Extracting text..."):
            text, extracted_links = parse_uploaded_file(uploaded)

        if not text or text.startswith("[ERROR"):
            st.error(text or "No text extracted")
            st.stop()

        with st.expander("Extracted text preview"):
            st.text_area("Text", text, height=220)

        if extracted_links:
            with st.expander(f"Extracted Links ({len(extracted_links)})"):
                for link in extracted_links:
                    st.markdown(f"- {link}")

        if st.button("Step 3: Run Extraction", type="primary"):
            with st.spinner("Running model..."):
                entity = extract_entities(
                    text,
                    provider=provider,
                    ollama_model=ollama_model,
                    ollama_host=ollama_host,
                    selected_fields=selected_fields,
                    extracted_links=extracted_links,
                    truncation_limit=len(text),
                )

            if entity is None:
                st.error("Extraction failed. Check local Ollama host/model.")
                st.stop()

            data = entity.model_dump()
            st.success("Extraction complete")

            st.markdown("### Editable Output")
            edited = {}
            with st.form("edit_form"):
                for field in selected_fields:
                    label = FIELD_SPECS[field]["label"]
                    value = data.get(field)
                    if field in LIST_FIELDS:
                        edited[field] = st.text_area(label, ", ".join(value or []), height=80)
                    else:
                        edited[field] = st.text_input(label, value or "")
                submit = st.form_submit_button("Save")

            if submit:
                payload = {}
                for field in selected_fields:
                    if field in LIST_FIELDS:
                        payload[field] = parse_list_input(edited[field])
                    else:
                        payload[field] = edited[field].strip() or None
                entity = ResumeEntity(**payload)
                data = entity.model_dump()
                st.success("Validated")

            filtered = {k: data.get(k) for k in selected_fields}
            st.json(filtered)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download JSON",
                    data=json.dumps(filtered, indent=2, ensure_ascii=False),
                    file_name="resume_parsed.json",
                    mime="application/json",
                )
            with col2:
                csv_payload = pd.DataFrame([entity_to_row(entity, selected_fields)]).to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv_payload,
                    file_name="resume_parsed.csv",
                    mime="text/csv",
                )

elif mode == "Evaluate Quality":
    st.subheader("Evaluate Extraction Quality")
    st.caption("Runs local Ollama model on labeled samples and reports precision/recall/F1.")
    gt_path = Path("Entity Recognition in Resumes.json")
    if not gt_path.exists():
        st.error("Entity Recognition in Resumes.json not found")
        st.stop()

    samples = st.slider("Samples", min_value=1, max_value=10, value=5, step=1)

    if st.button("Run Evaluation", type="primary"):
        raw = load_ground_truth(str(gt_path))[:samples]
        predictions = []
        ground_truths = []

        progress = st.progress(0)
        for index, sample in enumerate(raw, start=1):
            progress.progress(index / len(raw), text=f"Evaluating {index}/{len(raw)}")
            content = sample.get("content", "")
            annotations = sample.get("annotation", [])
            if not content or not annotations:
                continue

            gold = annotations_to_entity(annotations)
            pred = extract_entities(
                content,
                ollama_model=ollama_model,
                ollama_host=ollama_host,
            )
            if pred is None:
                pred = ResumeEntity()

            predictions.append(pred)
            ground_truths.append(gold)
            time.sleep(0.05)

        if not predictions:
            st.error("Evaluation failed: no predictions generated")
            st.stop()

        metrics = evaluate_batch(predictions, ground_truths)
        df_metrics = results_to_dataframe(metrics)

        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = runs_dir / f"evaluation_{stamp}.json"
        df_path = runs_dir / f"evaluation_{stamp}.csv"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        df_metrics.to_csv(df_path, index=False)

        macro = metrics["macro_avg"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Macro Precision", f"{macro['precision']:.3f}")
        c2.metric("Macro Recall", f"{macro['recall']:.3f}")
        c3.metric("Macro F1", f"{macro['f1']:.3f}")

        st.dataframe(df_metrics, use_container_width=True)
        st.info(f"Saved evaluation report to: {report_path}")
        st.info(f"Saved evaluation table to: {df_path}")
        st.download_button(
            "Download evaluation report",
            data=json.dumps(metrics, indent=2),
            file_name="evaluation_report.json",
            mime="application/json",
        )


