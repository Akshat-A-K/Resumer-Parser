# Fine-Tuning Guide (Student Friendly)

This project now has a simple local fine-tuning pipeline.

It also includes:
- Local Ollama resume-parser model specialization from Streamlit

## Why two datasets are used

- `Entity Recognition in Resumes.json`:
  - Labeled NER data (used for supervised fine-tuning)
  - Split into 80/10/10 train/val/test
- `Resume/Resume.csv`:
  - Unlabeled resume text
  - Exported for domain text usage (`splits/resume_unlabeled.txt`)

## Recommended model

For Python 3.14 in this project:
- `en_core_web_sm` or blank spaCy NER (stable install)

For best possible transformer accuracy:
- use Python 3.11/3.12 in a separate environment and then try `en_core_web_trf`
- reason: `spacy-transformers` currently fails to build on Python 3.14 in this setup

## Run steps

1. Prepare splits (80/10/10):

```bash
python finetune_prepare.py
```

2. Train model:

```bash
python finetune_train.py
```

3. Evaluate on test split:

```bash
python finetune_evaluate.py
```

## Streamlit handling (recommended)

Open Fine-tuning mode in the app and run in this order:
1. Install Training Dependencies
2. Download en_core_web_sm
3. Prepare Split
4. Train Model
5. Evaluate Model

## Local llama specialization from Streamlit

In Fine-tuning mode:
1. Set Ollama base model (example: llama3.2:3b)
2. Set new model name (example: resume-parser-local)
3. Click Create Local Resume Llama Model

Then use provider `ollama-local` and set OLLAMA_MODEL to the new name.

## Expected outputs

- `splits/train.jsonl`
- `splits/val.jsonl`
- `splits/test.jsonl`
- `splits/resume_unlabeled.txt`
- `models/resume_ner/`

## Notes

- Fine-tuning quality mostly depends on labeled NER data size/quality.
- Resume.csv has no entity labels, so it cannot directly replace labeled NER training.
- You can still use Resume.csv text for domain adaptation experiments.
- This repository is configured to run end-to-end on Python 3.14 without transformer extras.
- Training crash `[E955]` is handled by using `resume_training()` for pre-trained spaCy models.
