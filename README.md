# Resume Entity Extraction (Assignment 3)

Streamlit app for section-wise resume entity extraction and quick quality check.

## Streamlit page

- What it does: extracts resume entities into JSON and lets you review/edit results.
- How it works: two modes.
   - Parse One Resume: select fields, upload PDF/image, run extraction, edit/save JSON.
   - Evaluate Quality: compare extraction quality on labeled samples with Precision/Recall/F1.

## Workflow (implementation)

- Input handling: the UI accepts PDF or image files and reads bytes in memory.
- OCR and link capture: text is extracted, and URLs are detected and classified (LinkedIn, GitHub, etc.).
- Field selection: the user chooses which schema fields to extract; only those keys are allowed.
- Prompt building: a strict JSON template and annotated schema are injected into the LLM prompt.
- LLM inference: local Ollama is called with deterministic settings to return JSON only.
- Parsing and repair: JSON is parsed, lightly repaired if needed, and extra keys are dropped.
- Normalization: strings, lists, dates, and phone numbers are cleaned to match schema types.
- Validation: Pydantic schema validates fields; structured sections are kept consistent.
- Caching: results are cached by hash of (text + model + fields + truncation).
- Evaluation: labeled data is mapped to schema fields and Precision/Recall/F1 is computed.

## Setup

1. Create venv:
   - `python -m venv .venv`
2. Activate:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
3. Install deps:
   - `pip install -r requirements.txt`
4. Download model:
   - `ollama pull llama3.2`

## Run app

- Start UI: `streamlit run app.py`
- Open the link shown in terminal.

## Data and results

- Labels: [Entity Recognition in Resumes.json](Entity%20Recognition%20in%20Resumes.json)
- Resume samples: [Resume/Resume.csv](Resume/Resume.csv)
- Latest evaluation output: [results/](results/)
