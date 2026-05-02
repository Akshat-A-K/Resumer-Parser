# T13.1 smart document parser streamlit app

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluator import annotations_to_entity, evaluate_batch, load_ground_truth, results_to_dataframe
from llm_parser import extract_entities, get_last_error
from ocr_extractor import (
    extract_text_and_links_from_bytes,
    extract_text_from_image_bytes,
    extract_urls_from_text,
)
from schema import (
    FIELD_SPECS, LIST_FIELDS, STRUCTURED_FIELDS,
    ResumeEntity, default_selected_fields,
)

load_dotenv()

st.set_page_config(page_title="Resume Parser", page_icon="", layout="wide")
st.title("Smart Document Parser")


def parse_uploaded_file(uploaded_file) -> tuple[str, list[str]]:
    """Parse uploaded file and return (extracted_text, extracted_links)."""
    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.read()

    if suffix == ".pdf":
        text, links = extract_text_and_links_from_bytes(payload)
        return text, links
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        text = extract_text_from_image_bytes(payload)
        links = extract_urls_from_text(text) if text else []
        return text, links
    return "[ERROR: Unsupported file format]", []


def entity_to_row(entity, selected_fields):
    data = entity.model_dump()
    row = {}
    for field in selected_fields:
        value = data.get(field)
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                row[field] = json.dumps(value, ensure_ascii=False)
            else:
                row[field] = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            row[field] = json.dumps(value, ensure_ascii=False)
        else:
            row[field] = value
    return row


def parse_list_input(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def _classify_link(url: str) -> str:
    """Classify a URL into a human-readable category."""
    url_lower = url.lower().rstrip("/")

    # LinkedIn
    if "linkedin.com" in url_lower:
        return "LinkedIn Profile"

    # GitHub  profile vs project
    if "github.com" in url_lower:
        # Extract path parts after github.com
        import re
        match = re.search(r'github\.com/([^/?#]+)(?:/([^/?#]+))?', url_lower)
        if match:
            username = match.group(1)
            repo = match.group(2)
            # Skip github special pages
            if username in ("features", "marketplace", "explore", "topics", "settings"):
                return "GitHub"
            if repo and repo not in ("", "repositories", "stars", "followers", "following"):
                return "GitHub Project"
            return "GitHub Profile"
        return "GitHub"

    # Twitter/X
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "Twitter/X Profile"

    # Kaggle
    if "kaggle.com" in url_lower:
        return "Kaggle Profile"

    # Stack Overflow
    if "stackoverflow.com" in url_lower or "stackexchange.com" in url_lower:
        return "StackOverflow"

    # Medium
    if "medium.com" in url_lower:
        return "Blog"

    # Portfolio / personal sites
    if any(kw in url_lower for kw in ("portfolio", "personal", ".me", ".dev", "about.me")):
        return "Portfolio"

    # LeetCode / competitive programming
    if "leetcode.com" in url_lower:
        return "LeetCode Profile"
    if "codeforces.com" in url_lower:
        return "Codeforces Profile"
    if "codechef.com" in url_lower:
        return "CodeChef Profile"

    # Mailto
    if url_lower.startswith("mailto:"):
        return "Email"

    return "Link"

def _find_ollama_cli():
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
def get_ollama_model_options(current_model):
    # only show models actually installed in ollama
    options = []
    ollama_cli = _find_ollama_cli()

    if ollama_cli:
        try:
            # call ollama with a short timeout so the UI doesn't hang if the CLI blocks
            try:
                result = subprocess.run([ollama_cli, "list"], capture_output=True, text=True, timeout=3)
            except subprocess.TimeoutExpired:
                result = None

            if result and result.returncode == 0 and result.stdout:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                # skip header line if present
                for line in lines[1:]:
                    model_name = line.split()[0]
                    if model_name not in options:
                        options.append(model_name)
        except Exception:
            pass

    # add current_model if not already in list
    if current_model and current_model not in options:
        options.append(current_model)

    if not options:
        options = [current_model or "llama3.2:3b"]

    return options


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    st.subheader("Model Selection")
    model_choice = st.radio(
        "Choose model",
        ["ollama-basic", "ollama-finetuned"],
        format_func=lambda x: {
            "ollama-basic": "Ollama (Basic)",
            "ollama-finetuned": "Ollama (Finetuned)",
        }[x],
        index=0,
    )
    provider = "ollama-local"
    finetuned = "finetuned" in model_choice

    st.markdown("---")

    # initialize all provider vars to prevent NameErrors
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    if provider == "ollama-local":
        ollama_model_options = get_ollama_model_options(ollama_model)
        if finetuned:
            ollama_model = "resume-parser-local-3b"
        else:
            if ollama_model not in ollama_model_options and ollama_model_options:
                ollama_model = ollama_model_options[0]

    truncation_limit = 6000
    
    st.subheader("Workflow")
    mode = st.radio("Choose task", ["Parse One Resume", "Evaluate Quality"], index=0)


# ---------------------------------------------------------------------------
# Helper: render structured data (education/experience/projects)
# ---------------------------------------------------------------------------

def render_structured_field(field_name: str, entries: list):
    """Render structured entries as expandable cards."""
    if not entries:
        st.info("No entries found.")
        return

    if field_name == "education_details":
        for i, entry in enumerate(entries):
            e = entry if isinstance(entry, dict) else entry.model_dump()
            header = e.get("degree", "Education") or "Education"
            inst = e.get("institution", "")
            if inst:
                header = f"{header}  {inst}"
            with st.expander(f" {header}", expanded=(i == 0)):
                cols = st.columns(2)
                cols[0].markdown(f"**Institution:** {e.get('institution') or ''}")
                cols[0].markdown(f"**Degree:** {e.get('degree') or ''}")
                cols[0].markdown(f"**Field of Study:** {e.get('field_of_study') or ''}")
                cols[1].markdown(f"**CGPA/GPA:** {e.get('cgpa') or ''}")
                cols[1].markdown(f"**Location:** {e.get('location') or ''}")
                period = f"{e.get('start_year', '?')}  {e.get('end_year', '?')}"
                cols[1].markdown(f"**Period:** {period}")

    elif field_name == "experience_details":
        for i, entry in enumerate(entries):
            e = entry if isinstance(entry, dict) else entry.model_dump()
            header = e.get("role", "Experience") or "Experience"
            comp = e.get("company", "")
            if comp:
                header = f"{header} @ {comp}"
            with st.expander(f" {header}", expanded=(i == 0)):
                cols = st.columns(2)
                cols[0].markdown(f"**Company:** {e.get('company') or ''}")
                cols[0].markdown(f"**Role:** {e.get('role') or ''}")
                cols[0].markdown(f"**Location:** {e.get('location') or ''}")
                period = f"{e.get('start_date', '?')}  {e.get('end_date', '?')}"
                cols[1].markdown(f"**Period:** {period}")
                tech = e.get("tech_stack", [])
                if tech:
                    cols[1].markdown(f"**Tech Stack:** {', '.join(tech)}")
                desc = e.get("description")
                if desc:
                    st.markdown(f"**Description:** {desc}")

    elif field_name == "projects_detailed":
        for i, entry in enumerate(entries):
            e = entry if isinstance(entry, dict) else entry.model_dump()
            header = e.get("name", "Project") or "Project"
            with st.expander(f" {header}", expanded=(i == 0)):
                cols = st.columns(2)
                cols[0].markdown(f"**Name:** {e.get('name') or ''}")
                tech = e.get("tech_stack", [])
                if tech:
                    cols[0].markdown(f"**Tech Stack:** {', '.join(tech)}")
                link = e.get("link")
                if link:
                    link_str = str(link).strip()
                    if link_str.lower().startswith(("http://", "https://")):
                        label = _classify_link(link_str)
                        cols[1].markdown(f"**Link:** [{label}]({link_str})")
                    elif "." in link_str and " " not in link_str:
                        href = "https://" + link_str
                        label = _classify_link(href)
                        cols[1].markdown(f"**Link:** [{label}]({href})")
                    else:
                        cols[1].markdown(f"**Link:** {link_str}")
                date = e.get("date")
                if date:
                    cols[1].markdown(f"**Date:** {date}")
                desc = e.get("description")
                if desc:
                    st.markdown(f"**Description:** {desc}")


# ---------------------------------------------------------------------------
# Parse One Resume
# ---------------------------------------------------------------------------

if mode == "Parse One Resume":
    st.subheader("Parse One Resume")
    st.caption("Step 1: choose fields. Step 2: upload resume. Step 3: run extraction and edit results.")

    if "selected_fields" not in st.session_state:
        st.session_state["selected_fields"] = default_selected_fields()

    all_fields = list(FIELD_SPECS.keys())
    selected_fields = st.multiselect(
        f"Select fields to extract ({len(all_fields)} available)",
        options=all_fields,
        default=st.session_state["selected_fields"],
        key="field_multiselect",
        format_func=lambda x: FIELD_SPECS[x]["label"],
    )
    # update session state so presets work
    st.session_state["selected_fields"] = selected_fields

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

        char_count = len(text)
        with st.expander(f"Extracted text preview ({char_count} chars)", expanded=False):
            if char_count > 7000:
                st.warning("Note: This document is quite long. Large documents may take several minutes to parse with local models.")
            st.text_area("Text", text, height=220)

        # show extracted links with smart classification
        if extracted_links:
            with st.expander(f"Extracted Links ({len(extracted_links)})", expanded=False):
                for link in extracted_links:
                    label = _classify_link(link)
                    st.markdown(f"- **{label}:** [{link}]({link})")

        if st.button("Step 3: Run Extraction", type="primary"):
            mode_label = "Finetuned" if finetuned else "Direct"
            provider_label = "Ollama"
            with st.spinner(f"Running {provider_label} ({mode_label})..."):
                    t0 = time.time()
                    effective_truncation = len(text)
                    entity = extract_entities(
                        text,
                        provider=provider,
                        ollama_model=ollama_model,
                        ollama_host=ollama_host,
                        selected_fields=selected_fields,
                        finetuned=finetuned,
                        extracted_links=extracted_links,
                        truncation_limit=effective_truncation,
                    )
                    duration = time.time() - t0

            if entity is None:
                err = get_last_error()
                st.error(f"Extraction failed: {err}" if err else "Extraction failed. Check model settings.")
                st.stop()

            # display duration if available
            if 'duration' in locals():
                st.info(f"Extraction time: {duration:.2f}s")

            # store in session state so download buttons survive reruns
            st.session_state["last_entity"] = entity
            st.session_state["last_selected_fields"] = selected_fields
            st.session_state["last_provider"] = provider_label
            st.session_state["last_mode"] = mode_label

        # render results from session state
        if "last_entity" in st.session_state:
            entity = st.session_state["last_entity"]
            selected_fields_render = st.session_state.get("last_selected_fields", selected_fields)
            data = entity.model_dump()
            provider_label = st.session_state.get("last_provider", "")
            mode_label = st.session_state.get("last_mode", "")

            st.success(f"Extraction complete ({provider_label} {mode_label})")

            # --- Structured data display ---
            structured_in_selection = [f for f in STRUCTURED_FIELDS if f in selected_fields_render]
            if "skills_categorized" in data and data["skills_categorized"]:
                st.write("####  Skills by Category")
                cols = st.columns(2)
                idx = 0
                for cat, skills in data["skills_categorized"].items():
                    if skills:
                        with cols[idx % 2]:
                            st.markdown(f"**{cat}**")
                            st.write(", ".join(skills))
                        idx += 1
                st.markdown("---")
            if structured_in_selection:
                st.markdown("### Structured Data")
                tabs = st.tabs([FIELD_SPECS[f]["label"] for f in structured_in_selection])
                for tab, field in zip(tabs, structured_in_selection):
                    with tab:
                        entries = data.get(field, [])
                        render_structured_field(field, entries)

            # --- Flat fields editable form ---
            flat_fields = [f for f in selected_fields_render if f not in STRUCTURED_FIELDS]
            if flat_fields:
                st.markdown("### Editable Output")
                edited = {}
                with st.form("edit_form"):
                    for field in flat_fields:
                        label = FIELD_SPECS[field]["label"]
                        value = data.get(field)
                        if field in LIST_FIELDS:
                            edited[field] = st.text_area(label, ", ".join(value or []), height=80)
                        else:
                            edited[field] = st.text_input(label, value or "")
                    submit = st.form_submit_button("Save")

                if submit:
                    payload = {}
                    for field in flat_fields:
                        if field in LIST_FIELDS:
                            payload[field] = parse_list_input(edited[field])
                        else:
                            payload[field] = edited[field].strip() or None
                    # preserve structured fields
                    for field in structured_in_selection:
                        payload[field] = data.get(field, [])
                    entity = ResumeEntity(**payload)
                    data = entity.model_dump()
                    st.session_state["last_entity"] = entity
                    st.success("Validated & saved")

            # --- JSON view ---
            filtered = {k: data.get(k) for k in selected_fields_render}
            st.json(filtered)

            # --- Download buttons (always visible because data is in session_state) ---
            json_str = json.dumps(filtered, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                " Download JSON",
                data=json_str,
                file_name="resume_parsed.json",
                mime="application/json",
                key="download_json",
            )


# ---------------------------------------------------------------------------
# Evaluate Quality
# ---------------------------------------------------------------------------

elif mode == "Evaluate Quality":
    st.subheader("Evaluate Extraction Quality")
    st.caption("Compare multiple configurations on labeled samples and reports precision/recall/F1.")

    gt_path = Path("Entity Recognition in Resumes.json")
    if not gt_path.exists():
        st.error("Entity Recognition in Resumes.json not found")
        st.stop()

    samples = st.slider("Samples", min_value=5, max_value=220, value=10, step=5)
    
    eval_configs = ["ollama-direct", "ollama-finetuned"]
    st.caption("Evaluates Ollama direct vs finetuned")

    if st.button("Run Evaluation", type="primary"):
        if not eval_configs:
            st.error("Select at least one configuration")
            st.stop()

        raw = load_ground_truth(str(gt_path))[:samples]
        ground_truths = []
        for sample in raw:
            annotations = sample.get("annotation", [])
            if annotations:
                ground_truths.append(annotations_to_entity(annotations))

        if not ground_truths:
            st.error("No valid ground truth samples found.")
            st.stop()

        all_config_metrics = {}
        
        for config_key in eval_configs:
            is_ft = "finetuned" in config_key
            label = config_key.replace("-", " ").title()
            
            st.write(f"### Evaluating {label}...")
            predictions = []
            progress = st.progress(0)
            
            for index, sample in enumerate(raw, start=1):
                progress.progress(index / len(raw), text=f"Processing {index}/{len(raw)}")
                content = sample.get("content", "")
                if not content:
                    continue

                effective_truncation = len(content)
                pred = extract_entities(
                    content,
                    provider="ollama-local",
                    ollama_model=ollama_model,
                    ollama_host=ollama_host,
                    finetuned=is_ft,
                    truncation_limit=effective_truncation,
                )
                if pred is None:
                    pred = ResumeEntity()
                predictions.append(pred)
                # larger sleep for batch mode to respect rate limits
                time.sleep(0.1)

            metrics = evaluate_batch(predictions, ground_truths)
            all_config_metrics[config_key] = metrics
            
            df_metrics = results_to_dataframe(metrics)
            st.dataframe(df_metrics, use_container_width=True)

        if len(all_config_metrics) > 1:
            st.write("### Comparison Summary")
            comparison_rows = []
            for cfg, m in all_config_metrics.items():
                macro = m["macro_avg"]
                comparison_rows.append({
                    "Config": cfg.replace("-", " ").title(),
                    "Precision": round(macro["precision"], 3),
                    "Recall": round(macro["recall"], 3),
                    "F1 Score": round(macro["f1"], 3)
                })
            st.table(pd.DataFrame(comparison_rows))


