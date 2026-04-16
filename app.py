"""T13.1 Smart Document Parser - Streamlit app."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluator import annotations_to_entity, evaluate_batch, load_ground_truth, results_to_dataframe
from llm_parser import extract_entities
from ocr_extractor import extract_text_from_bytes, extract_text_from_image_bytes
from schema import FIELD_SPECS, LIST_FIELDS, ResumeEntity, default_selected_fields

load_dotenv()

st.set_page_config(page_title="T13.1 Resume Parser", page_icon="📄", layout="wide")
st.title("T13.1 - Smart Document Parser")


def parse_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.read()

    if suffix == ".pdf":
        return extract_text_from_bytes(payload)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        return extract_text_from_image_bytes(payload)
    return "[ERROR: Unsupported file format]"


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


with st.sidebar:
    st.header("Workflow")
    mode = st.radio(
        "Choose task",
        ["Parse One Resume", "Evaluate Quality", "Fine-tune Models"],
        index=0,
    )

    st.markdown("---")
    st.header("Model Provider")
    provider = st.selectbox("Provider", ["gemini", "groq", "ollama-local", "openai"], index=0)

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    st.caption("Only the selected provider settings are shown by default.")
    if provider == "gemini":
        gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=gemini_key)
    elif provider == "groq":
        groq_key = st.text_input("GROQ_API_KEY", type="password", value=groq_key)
    elif provider == "ollama-local":
        ollama_host = st.text_input("OLLAMA_HOST", value=ollama_host)
        ollama_model = st.text_input("OLLAMA_MODEL", value=ollama_model)
    elif provider == "openai":
        openai_key = st.text_input("OPENAI_API_KEY", type="password", value=openai_key)
        openai_model = st.text_input("OPENAI_MODEL", value=openai_model)

    with st.expander("Advanced provider settings", expanded=False):
        gemini_key = st.text_input("GEMINI_API_KEY (advanced)", type="password", value=gemini_key)
        groq_key = st.text_input("GROQ_API_KEY (advanced)", type="password", value=groq_key)
        ollama_host = st.text_input("OLLAMA_HOST (advanced)", value=ollama_host)
        ollama_model = st.text_input("OLLAMA_MODEL (advanced)", value=ollama_model)
        openai_key = st.text_input("OPENAI_API_KEY (advanced)", type="password", value=openai_key)
        openai_model = st.text_input("OPENAI_MODEL (advanced)", value=openai_model)


def run_python_script(script_name: str, args: list[str] | None = None) -> tuple[bool, str]:
    """Run project script with current Python environment and capture output."""
    script_path = Path(script_name)
    if not script_path.exists():
        return False, f"Script not found: {script_name}"

    result = subprocess.run(
        [sys.executable, script_name] + (args or []),
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, combined.strip()


def run_command(args: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, combined.strip()

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
            text = parse_uploaded_file(uploaded)

        if not text or text.startswith("[ERROR"):
            st.error(text or "No text extracted")
            st.stop()

        with st.expander("Extracted text preview"):
            st.text_area("Text", text, height=220)

        if st.button("Step 3: Run Extraction", type="primary"):
            with st.spinner("Running model..."):
                entity = extract_entities(
                    text,
                    provider=provider,
                    gemini_key=gemini_key,
                    groq_key=groq_key,
                    ollama_model=ollama_model,
                    ollama_host=ollama_host,
                    openai_key=openai_key,
                    openai_model=openai_model,
                    selected_fields=selected_fields,
                )

            if entity is None:
                st.error("Extraction failed. Check provider/API key or local Ollama host/model.")
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
    st.caption("Runs your selected provider on labeled samples and reports precision/recall/F1.")
    gt_path = Path("Entity Recognition in Resumes.json")
    if not gt_path.exists():
        st.error("Entity Recognition in Resumes.json not found")
        st.stop()

    samples = st.slider("Samples", min_value=5, max_value=220, value=20, step=5)

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
                provider=provider,
                gemini_key=gemini_key,
                groq_key=groq_key,
                ollama_model=ollama_model,
                ollama_host=ollama_host,
                openai_key=openai_key,
                openai_model=openai_model,
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

        macro = metrics["macro_avg"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Macro Precision", f"{macro['precision']:.3f}")
        c2.metric("Macro Recall", f"{macro['recall']:.3f}")
        c3.metric("Macro F1", f"{macro['f1']:.3f}")

        st.dataframe(df_metrics, use_container_width=True)
        st.download_button(
            "Download evaluation report",
            data=json.dumps(metrics, indent=2),
            file_name="evaluation_report.json",
            mime="application/json",
        )

else:
    st.subheader("Fine-tune Models")
    st.caption("Use one tab at a time based on your goal: local spaCy model, local Ollama profile, or OpenAI API fine-tuning.")

    def show_step(script_name: str, title: str) -> bool:
        with st.spinner(f"Running {script_name}..."):
            ok, output = run_python_script(script_name)
        if ok:
            st.success(f"{title} completed")
            if output:
                st.text_area(f"{title} output", output, height=180)
        else:
            st.error(f"{title} failed")
            if output:
                st.text_area(f"{title} error output", output, height=220)
        return ok

    tab_local, tab_ollama, tab_api = st.tabs(
        ["Local spaCy NER", "Local Ollama Profile", "OpenAI API Fine-tune"]
    )

    with tab_local:
        st.markdown("### Local spaCy Pipeline")
        st.write("1. Prepare split -> 2. Train -> 3. Evaluate")

        with st.expander("One-time setup (if needed)", expanded=False):
            s1, s2 = st.columns(2)
            setup_deps = s1.button("Install Training Dependencies", key="setup_deps")
            setup_model = s2.button("Download en_core_web_sm", key="setup_model")

            if setup_deps:
                with st.spinner("Installing dependencies from requirements.txt..."):
                    ok, output = run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                if ok:
                    st.success("Dependencies installed")
                else:
                    st.error("Dependency installation failed")
                if output:
                    st.text_area("Dependency setup output", output, height=220, key="deps_output")

            if setup_model:
                with st.spinner("Downloading en_core_web_sm..."):
                    ok, output = run_command([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                if ok:
                    st.success("en_core_web_sm downloaded")
                else:
                    st.error("Model download failed")
                if output:
                    st.text_area("Model setup output", output, height=220, key="model_output")

        c1, c2, c3, c4 = st.columns(4)
        run_prepare = c1.button("Prepare Split", type="primary", key="run_prepare")
        run_train = c2.button("Train Model", key="run_train")
        run_eval = c3.button("Evaluate Model", key="run_eval")
        run_all = c4.button("Run All", key="run_all")

        if run_prepare:
            show_step("finetune_prepare.py", "Prepare split")

        if run_train:
            show_step("finetune_train.py", "Train model")

        if run_eval:
            show_step("finetune_evaluate.py", "Evaluate model")

        if run_all:
            ok_prepare = show_step("finetune_prepare.py", "Prepare split")
            if ok_prepare:
                ok_train = show_step("finetune_train.py", "Train model")
                if ok_train:
                    show_step("finetune_evaluate.py", "Evaluate model")

    with tab_ollama:
        st.markdown("### Local Llama Specialization")
        st.caption("Creates a resume-focused local model profile from your base Ollama model.")
        llama_base = st.text_input("Ollama base model", value=ollama_model)
        llama_new = st.text_input("New local model name", value="resume-parser-local")
        run_llama_tune = st.button("Create Local Resume Llama Model", key="run_llama_tune")

        if run_llama_tune:
            with st.spinner("Creating local Ollama model..."):
                ok, output = run_python_script(
                    "llama_local_tune.py",
                    ["--base-model", llama_base, "--new-model", llama_new],
                )
            if ok:
                st.success(f"Created local model: {llama_new}")
                st.info("Set provider = ollama-local and OLLAMA_MODEL to this new model.")
            else:
                st.error("Local llama specialization failed")
            if output:
                st.text_area("Local llama output", output, height=220, key="llama_output")

    with tab_api:
        st.markdown("### OpenAI API Fine-tuning")
        st.caption("Prepare dataset, start a fine-tune job, and check job status.")

        api_samples = st.number_input(
            "API tuning max samples",
            min_value=50,
            max_value=2000,
            value=250,
            step=50,
        )
        api_base_model = st.text_input("API base model", value=openai_model, key="api_base_model")
        api_job_id = st.text_input("Existing fine-tune job id (for status)", value="")

        a1, a2, a3 = st.columns(3)
        api_prepare = a1.button("Prepare API Dataset", key="api_prepare")
        api_start = a2.button("Start API Fine-tune", key="api_start")
        api_status = a3.button("Check API Job Status", key="api_status")

        if api_prepare:
            with st.spinner("Preparing API dataset..."):
                ok, output = run_python_script("api_finetune.py", ["prepare", "--max-samples", str(api_samples)])
            if ok:
                st.success("API dataset prepared")
            else:
                st.error("API dataset preparation failed")
            if output:
                st.text_area("API prepare output", output, height=180, key="api_prepare_output")

        if api_start:
            if not openai_key:
                st.error("Set OPENAI_API_KEY in sidebar first")
            else:
                with st.spinner("Starting API fine-tuning job..."):
                    ok, output = run_python_script(
                        "api_finetune.py",
                        ["start", "--api-key", openai_key, "--base-model", api_base_model],
                    )
                if ok:
                    st.success("API fine-tuning job started")
                else:
                    st.error("API fine-tuning start failed")
                if output:
                    st.text_area("API start output", output, height=220, key="api_start_output")

        if api_status:
            if not openai_key:
                st.error("Set OPENAI_API_KEY in sidebar first")
            elif not api_job_id.strip():
                st.error("Enter fine-tune job id")
            else:
                with st.spinner("Checking API fine-tuning status..."):
                    ok, output = run_python_script(
                        "api_finetune.py",
                        ["status", "--api-key", openai_key, "--job-id", api_job_id.strip()],
                    )
                if ok:
                    st.success("Fetched API job status")
                else:
                    st.error("Failed to fetch API job status")
                if output:
                    st.text_area("API status output", output, height=260, key="api_status_output")
