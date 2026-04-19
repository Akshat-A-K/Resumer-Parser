# T13.1 smart document parser streamlit app

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import time

from evaluator import annotations_to_entity, evaluate_batch, load_ground_truth, results_to_dataframe
from llm_parser import clear_cache, extract_entities, get_cache_stats, get_last_error
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

st.set_page_config(page_title="T13.1 Resume Parser", page_icon="", layout="wide")
st.title("T13.1 Smart Document Parser")


# ---------------------------------------------------------------------------
# File parsing (returns text + links)
# ---------------------------------------------------------------------------

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

    # GitHub — profile vs project
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


def _selected_index(options, value):
    return options.index(value) if value in options else 0


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


def run_python_script(script_name, args=None):
    script_path = Path(script_name)
    if not script_path.exists():
        return False, f"Script not found: {script_name}"

    result = subprocess.run(
        [sys.executable, script_name] + (args or []),
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent,
    )
    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, combined.strip()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    st.subheader("Model Provider")
    provider = st.radio(
        "Choose provider",
        ["ollama-local", "gemini-api"],
        format_func=lambda x: "Ollama (Local)" if x == "ollama-local" else "Gemini (API)",
        index=0,
    )

    st.subheader("Extraction Mode")
    finetuned = st.toggle(
        "Use Finetuned Mode",
        value=False,
        help="When enabled, uses fewshot examples from training data for better results.",
    )

    st.markdown("---")

    # initialize all provider vars to prevent NameErrors
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if provider == "ollama-local":
        st.subheader("Ollama Settings")
        ollama_host = st.text_input("Host", value=ollama_host)
        ollama_model_options = get_ollama_model_options(ollama_model)

        # Offer quick creation of a local finetuned model (Modelfile-based few-shot)
        if st.button("Build local finetuned model (resume-parser-local)"):
            with st.spinner("Creating local Ollama model (may take a few minutes)..."):
                ok, out = run_python_script("finetune_ollama.py", args=["--base-model", ollama_model, "--new-model", "resume-parser-local"])
                if ok:
                    st.success("Local model created: resume-parser-local")
                else:
                    st.error(f"Failed to create model: {out}")

        # Warm the selected model in background to avoid long first-request delays
        try:
            import threading, requests

            def _warm_model_once(host, model):
                try:
                    url = host.rstrip("/") + "/api/chat"
                    payload = {
                        "model": model,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 16, "num_ctx": 512},
                        "messages": [{"role": "user", "content": "Say hi"}],
                    }
                    # long timeout here because loading may take time; run in background
                    requests.post(url, json=payload, timeout=120)
                except Exception:
                    pass

            if "_ollama_warmed" not in st.session_state and ollama_model:
                # start background thread to warm model
                t = threading.Thread(target=_warm_model_once, args=(ollama_host, ollama_model), daemon=True)
                t.start()
                st.session_state["_ollama_warmed"] = True
        except Exception:
            pass

        if finetuned:
            st.caption("Finetuned mode: using specialized model")
            ollama_model = st.selectbox(
                "Model", options=ollama_model_options,
                index=_selected_index(ollama_model_options, "resume-parser-local"),
            )
        else:
            ollama_model = st.selectbox(
                "Model", options=ollama_model_options,
                index=_selected_index(ollama_model_options, ollama_model),
            )
    else:
        st.subheader("Gemini Settings")
        gemini_api_key = st.text_input("API Key", value=gemini_api_key, type="password")
        gemini_model = st.text_input("Model", value=gemini_model)
        if finetuned:
            st.caption("Finetuned mode: using fewshot enhanced prompt")

    st.markdown("---")
    st.subheader("Performance Settings")
    truncation_limit = st.slider(
        "Text Truncation Limit",
        min_value=2000, max_value=12000, value=6000, step=1000,
        help="Higher = more context but slower. 4000-6000 is usually plenty."
    )
    
    st.subheader("Field Presets")
    c1, c2 = st.columns(2)
    if c1.button("Core Only"):
        st.session_state["selected_fields"] = ["name", "email", "phone", "skills_categorized", "location"]
        st.rerun()
    if c2.button("Reset Selection"):
        st.session_state["selected_fields"] = default_selected_fields()
        st.rerun()

    st.markdown("---")

    # cache controls
    st.subheader("Cache")
    cache_stats = get_cache_stats()
    st.caption(f"Cached results: **{cache_stats['count']}** ({cache_stats['size_mb']} MB)")
    if st.button("Clear Cache", key="clear_cache_btn"):
        removed = clear_cache()
        st.success(f"Cleared {removed} cached results")
        st.rerun()

    st.markdown("---")
    st.subheader("Workflow")
    mode = st.radio("Choose task", ["Parse One Resume", "Evaluate Quality", "Finetune Models"], index=0)


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
                header = f"{header} — {inst}"
            with st.expander(f"🎓 {header}", expanded=(i == 0)):
                cols = st.columns(2)
                cols[0].markdown(f"**Institution:** {e.get('institution') or '—'}")
                cols[0].markdown(f"**Degree:** {e.get('degree') or '—'}")
                cols[0].markdown(f"**Field of Study:** {e.get('field_of_study') or '—'}")
                cols[1].markdown(f"**CGPA/GPA:** {e.get('cgpa') or '—'}")
                cols[1].markdown(f"**Location:** {e.get('location') or '—'}")
                period = f"{e.get('start_year', '?')} – {e.get('end_year', '?')}"
                cols[1].markdown(f"**Period:** {period}")

    elif field_name == "experience_details":
        for i, entry in enumerate(entries):
            e = entry if isinstance(entry, dict) else entry.model_dump()
            header = e.get("role", "Experience") or "Experience"
            comp = e.get("company", "")
            if comp:
                header = f"{header} @ {comp}"
            with st.expander(f"💼 {header}", expanded=(i == 0)):
                cols = st.columns(2)
                cols[0].markdown(f"**Company:** {e.get('company') or '—'}")
                cols[0].markdown(f"**Role:** {e.get('role') or '—'}")
                cols[0].markdown(f"**Location:** {e.get('location') or '—'}")
                period = f"{e.get('start_date', '?')} – {e.get('end_date', '?')}"
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
            with st.expander(f"🔨 {header}", expanded=(i == 0)):
                cols = st.columns(2)
                cols[0].markdown(f"**Name:** {e.get('name') or '—'}")
                tech = e.get("tech_stack", [])
                if tech:
                    cols[0].markdown(f"**Tech Stack:** {', '.join(tech)}")
                link = e.get("link")
                if link:
                    cols[1].markdown(f"**Link:** [{link}]({link})")
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
            provider_label = "Ollama" if provider == "ollama-local" else "Gemini"
            with st.spinner(f"Running {provider_label} ({mode_label})..."):
                    t0 = time.time()
                    entity = extract_entities(
                        text,
                        provider=provider,
                        ollama_model=ollama_model,
                        ollama_host=ollama_host,
                        gemini_api_key=gemini_api_key,
                        gemini_model=gemini_model,
                        selected_fields=selected_fields,
                        finetuned=finetuned,
                        extracted_links=extracted_links,
                        truncation_limit=truncation_limit,
                    )
                    duration = time.time() - t0

            if entity is None:
                err = get_last_error()
                if "429" in str(err) or "quota" in str(err).lower():
                    st.error(f"Gemini Quota Exceeded. Please switch to **Ollama** in the sidebar to continue extraction without limits.")
                else:
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
                st.write("#### 🛠️ Skills by Category")
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
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_str,
                    file_name="resume_parsed.json",
                    mime="application/json",
                    key="download_json",
                )
            with col2:
                csv_payload = pd.DataFrame([entity_to_row(entity, selected_fields_render)]).to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_payload,
                    file_name="resume_parsed.csv",
                    mime="text/csv",
                    key="download_csv",
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
    
    eval_configs = st.multiselect(
        "Configurations to evaluate",
        options=["ollama-direct", "ollama-finetuned", "gemini-direct", "gemini-finetuned"],
        default=[f"{provider.replace('-local', '').replace('-api', '')}-{'finetuned' if finetuned else 'direct'}"],
        help="Select one or more configurations to compare."
    )

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
            is_gemini = "gemini" in config_key
            label = config_key.replace("-", " ").title()
            
            st.write(f"### Evaluating {label}...")
            predictions = []
            progress = st.progress(0)
            
            for index, sample in enumerate(raw, start=1):
                progress.progress(index / len(raw), text=f"Processing {index}/{len(raw)}")
                content = sample.get("content", "")
                if not content:
                    continue

                pred = extract_entities(
                    content,
                    provider="gemini-api" if is_gemini else "ollama-local",
                    ollama_model=ollama_model,
                    ollama_host=ollama_host,
                    gemini_api_key=gemini_api_key,
                    gemini_model=gemini_model,
                    finetuned=is_ft,
                )
                if pred is None:
                    pred = ResumeEntity()
                predictions.append(pred)
                # larger sleep for batch mode to respect rate limits
                time.sleep(1.0 if is_gemini else 0.1)

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


# ---------------------------------------------------------------------------
# Finetune Models
# ---------------------------------------------------------------------------

else:
    st.subheader("Finetune Models")
    st.caption("Prepare training data and create finetuned models for better extraction results.")

    def show_step(script_name, title, args=None):
        with st.spinner(f"Running {script_name}..."):
            ok, output = run_python_script(script_name, args)
        if ok:
            st.success(f"{title} completed")
            if output:
                st.text_area(f"{title} output", output, height=180)
        else:
            st.error(f"{title} failed")
            if output:
                st.text_area(f"{title} error output", output, height=220)
        return ok

    tab_prepare, tab_ollama, tab_gemini, tab_eval = st.tabs([
        "Prepare Data", "Finetune Ollama", "Finetune Gemini", "Evaluate Finetuned",
    ])

    with tab_prepare:
        st.markdown("### Prepare Training Data")
        st.write(
            "Converts the labeled dataset into training pairs (resume text to expected JSON output). "
            "Splits 80/10/10 with seed=42. Selects best fewshot examples."
        )

        col1, col2 = st.columns(2)
        src_ner = Path("Entity Recognition in Resumes.json")
        src_csv = Path("Resume/Resume.csv")
        src_pdf = Path("data/data")

        col1.markdown(f"**NER Dataset**: {'found' if src_ner.exists() else 'missing'}")
        col1.markdown(f"**Resume CSV**: {'found' if src_csv.exists() else 'missing'}")
        col2.markdown(f"**PDF Corpus**: {'found' if src_pdf.exists() else 'missing'}")

        splits_dir = Path("splits")
        if splits_dir.exists() and (splits_dir / "split_stats.json").exists():
            with (splits_dir / "split_stats.json").open("r") as f:
                stats = json.load(f)
            col2.markdown(f"**Existing splits**: Train={stats.get('train_count', '?')} | "
                         f"Val={stats.get('val_count', '?')} | Test={stats.get('test_count', '?')}")

        if st.button("Prepare Splits", type="primary", key="run_prepare"):
            show_step("finetune_prepare.py", "Data Preparation")

    with tab_ollama:
        st.markdown("### Finetune Ollama Model")
        st.write(
            "Creates a specialized Ollama model with a comprehensive system prompt "
            "and embedded fewshot examples from the training data."
        )

        ollama_model_options_ft = get_ollama_model_options(os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
        ft_base = st.selectbox(
            "Base model", options=ollama_model_options_ft,
            index=_selected_index(ollama_model_options_ft, "llama3.2:3b"),
            key="ft_ollama_base",
        )
        ft_name = st.text_input("New model name", value="resume-parser-local", key="ft_ollama_name")

        fewshot_path = Path("splits/fewshot_examples.json")
        if fewshot_path.exists():
            with fewshot_path.open("r") as f:
                fewshot_data = json.load(f)
            st.info(f"{len(fewshot_data)} fewshot examples available from training data")

        if st.button("Create Finetuned Ollama Model", type="primary", key="run_ollama_ft"):
            show_step("finetune_ollama.py", "Ollama Finetuning",
                     ["--base-model", ft_base, "--new-model", ft_name])

    with tab_gemini:
        st.markdown("### Finetune Gemini")
        st.write(
            "Prepares enhanced fewshot prompt for Gemini API. "
            "Tests extraction quality with the enhanced prompt."
        )

        gemini_key_ft = st.text_input(
            "Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""),
            type="password", key="ft_gemini_key",
        )
        gemini_model_ft = st.text_input(
            "Gemini Model", value=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            key="ft_gemini_model",
        )
        test_samples_ft = st.number_input("Test samples", min_value=1, max_value=10, value=3, key="ft_test_samples")

        if st.button("Validate Gemini Finetuning", type="primary", key="run_gemini_ft"):
            if not gemini_key_ft:
                st.error("Set GEMINI_API_KEY first")
            else:
                show_step("finetune_gemini.py", "Gemini Finetuning",
                         ["--test-samples", str(test_samples_ft), "--model", gemini_model_ft])

    with tab_eval:
        st.markdown("### Evaluate Finetuned Models")
        st.write("Run evaluation on test split using finetuned models and compare with direct mode.")

        test_path = Path("splits/test.jsonl")
        if not test_path.exists():
            st.warning("Test split not found. Run Prepare Data first.")
        else:
            with test_path.open("r") as f:
                test_count = sum(1 for line in f if line.strip())
            st.info(f"{test_count} test samples available")

        eval_samples = st.slider("Samples to evaluate", min_value=5, max_value=50, value=10, key="eval_samples")
        eval_configs = st.multiselect(
            "Configurations to evaluate",
            options=["ollama-direct", "ollama-finetuned", "gemini-direct", "gemini-finetuned"],
            default=["ollama-direct", "ollama-finetuned"],
            key="eval_configs",
        )

        if st.button("Run Evaluation", type="primary", key="run_ft_eval"):
            if not eval_configs:
                st.error("Select at least one configuration")
            else:
                configs_str = ",".join(eval_configs)
                show_step("finetune_evaluate.py", "Finetuned Evaluation",
                         ["--samples", str(eval_samples), "--configs", configs_str])
