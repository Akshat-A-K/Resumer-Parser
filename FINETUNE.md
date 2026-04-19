# Fine-Tuning Guide

This project uses LLM-based fine-tuning with two providers (Ollama local + Gemini API).

## Architecture

### Two Providers
- **Ollama (Local)**: Uses llama3.2:2b via local Ollama server
- **Gemini (API)**: Uses Google Gemini API (requires API key)

### Two Modes
- **Direct**: Original model with basic extraction prompt
- **Fine-tuned**: Enhanced model with few-shot examples from training data

## Datasets Used

| Dataset | Type | Usage |
|---------|------|-------|
| `Entity Recognition in Resumes.json` | NER-annotated (~220 samples) | Labeled training/validation/test (80/10/10) |
| `Resume/Resume.csv` | Unlabeled (2400+ resumes) | Domain corpus |
| `data/data/` | PDF resumes (2475 files, 24 categories) | Domain corpus |

## Run Steps

### 1. Prepare Training Data

```bash
python finetune_prepare.py
```

This creates:
- `splits/train.jsonl` — 80% training pairs
- `splits/val.jsonl` — 10% validation pairs
- `splits/test.jsonl` — 10% test pairs
- `splits/fewshot_examples.json` — Best examples for few-shot prompting
- `splits/corpus_unlabeled.txt` — Combined unlabeled text
- `splits/split_stats.json` — Statistics

### 2. Fine-tune Ollama Model

```bash
python finetune_ollama.py --base-model llama3.2:2b --new-model resume-parser-local
```

Creates a specialized Ollama model with:
- Comprehensive system prompt with all field definitions
- Few-shot examples embedded as conversation turns
- Optimal inference parameters (temperature=0, etc.)

### 3. Fine-tune Gemini (Validate)

```bash
python finetune_gemini.py --test-samples 3
```

Validates Gemini extraction with few-shot enhanced prompt.

### 4. Evaluate All Configurations

```bash
python finetune_evaluate.py --samples 20
```

Evaluates all 4 configurations and produces comparison report.

### 5. Compare on Full Dataset

```bash
python compare_parsers.py --samples 220 --mode all
```

## Streamlit App

```bash
streamlit run app.py
```

### Sidebar
- Select provider: Ollama (Local) or Gemini (API)
- Toggle fine-tuned mode on/off
- Choose specific model

### Workflows
1. **Parse One Resume** — Upload PDF/image → extract → edit → download
2. **Evaluate Quality** — Run on labeled samples, see P/R/F1
3. **Fine-tune Models** — Prepare data → Fine-tune Ollama → Fine-tune Gemini → Evaluate

## Configuration

Set in `.env`:
```
GEMINI_API_KEY=your_key_here
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:2b
GEMINI_MODEL=gemini-2.0-flash
```

## Expected Outputs

- `splits/` — Training splits and few-shot examples
- `api_tuning/Modelfile.resume` — Ollama Modelfile
- `runs/` — Evaluation reports (CSV + JSON)

## Schema Fields (25 total)

| Field | Type | Description |
|-------|------|-------------|
| name | string | Full name |
| email | string | Email address |
| phone | string | Phone number |
| location | string | City/state/address |
| linkedin | string | LinkedIn URL |
| github | string | GitHub URL |
| portfolio | string | Portfolio/website URL |
| twitter | string | Twitter/X URL |
| other_links | list | Other relevant URLs |
| designation | list | Job titles held |
| companies_worked_at | list | Company names |
| skills | list | Technical and soft skills |
| years_of_experience | string | Total experience |
| summary | string | Professional summary |
| college_name | list | Education institutions |
| degree | list | Degrees obtained |
| graduation_year | string | Graduation year |
| certifications | list | Professional certs |
| projects | list | Project names |
| languages | list | Human languages |
| achievements | list | Awards/honors |
| publications | list | Research/articles |
| hobbies | list | Interests |
| references | list | Professional references |
