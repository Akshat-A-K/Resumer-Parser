"""Compare local parser strategies: LLM-only, NER-only, and hybrid.

Usage:
    python compare_parsers.py --samples 220 --mode all --ollama-model resume-parser-local
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import spacy

from evaluator import annotations_to_entity, evaluate_batch, load_ground_truth, results_to_dataframe
from llm_parser import extract_entities
from schema import ResumeEntity

LABEL_TO_FIELD = {
    "NAME": "name",
    "EMAIL": "email",
    "LOCATION": "location",
    "GRAD_YEAR": "graduation_year",
    "EXPERIENCE": "years_of_experience",
    "DESIGNATION": "designation",
    "COMPANY": "companies_worked_at",
    "SKILL": "skills",
    "COLLEGE": "college_name",
    "DEGREE": "degree",
}

LIST_FIELDS = {"designation", "companies_worked_at", "skills", "college_name", "degree"}
SINGLE_FIELDS = {"name", "email", "location", "graduation_year", "years_of_experience"}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[\w\-/?=&.%]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-/?=&.%]+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare parser strategies on labeled dataset.")
    parser.add_argument("--samples", type=int, default=220)
    parser.add_argument("--dataset", default="Entity Recognition in Resumes.json")
    parser.add_argument("--mode", choices=["llm", "ner", "hybrid", "all"], default="all")
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--output-dir", default="runs")

    parser.add_argument("--spacy-model", default="models/resume_ner")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="resume-parser-local")
    return parser.parse_args()


def _pick_single(values: list[str]) -> str | None:
    if not values:
        return None
    values = [v.strip() for v in values if v and v.strip()]
    if not values:
        return None
    values.sort(key=lambda x: (-len(x), x.lower()))
    return values[0]


def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = (value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def ner_extract(nlp, text: str) -> ResumeEntity:
    doc = nlp(text)
    by_field: dict[str, list[str]] = {field: [] for field in LABEL_TO_FIELD.values()}

    for ent in doc.ents:
        field = LABEL_TO_FIELD.get(ent.label_)
        if field is None:
            continue
        value = ent.text.strip()
        if value:
            by_field[field].append(value)

    payload = {
        "name": _pick_single(by_field.get("name", [])),
        "email": _pick_single(by_field.get("email", [])),
        "location": _pick_single(by_field.get("location", [])),
        "graduation_year": _pick_single(by_field.get("graduation_year", [])),
        "years_of_experience": _pick_single(by_field.get("years_of_experience", [])),
        "designation": _unique(by_field.get("designation", [])),
        "companies_worked_at": _unique(by_field.get("companies_worked_at", [])),
        "skills": _unique(by_field.get("skills", [])),
        "college_name": _unique(by_field.get("college_name", [])),
        "degree": _unique(by_field.get("degree", [])),
    }

    return ResumeEntity(**payload)


def regex_enrich(entity: ResumeEntity, text: str) -> ResumeEntity:
    data = entity.model_dump()

    if not data.get("email"):
        hit = EMAIL_RE.search(text)
        if hit:
            data["email"] = hit.group(0)

    if not data.get("phone"):
        hit = PHONE_RE.search(text)
        if hit:
            data["phone"] = hit.group(0)

    if not data.get("linkedin"):
        hit = LINKEDIN_RE.search(text)
        if hit:
            data["linkedin"] = hit.group(0)

    if not data.get("github"):
        hit = GITHUB_RE.search(text)
        if hit:
            data["github"] = hit.group(0)

    return ResumeEntity(**data)


def merge_entities(llm_entity: ResumeEntity, ner_entity: ResumeEntity, text: str) -> ResumeEntity:
    llm = llm_entity.model_dump()
    ner = ner_entity.model_dump()

    merged = dict(llm)

    for field in SINGLE_FIELDS:
        if not merged.get(field) and ner.get(field):
            merged[field] = ner[field]

    for field in LIST_FIELDS:
        llm_vals = llm.get(field) or []
        ner_vals = ner.get(field) or []
        merged[field] = _unique([*llm_vals, *ner_vals])

    merged_entity = ResumeEntity(**merged)
    return regex_enrich(merged_entity, text)


def evaluate_mode(
    mode: str,
    rows: list[dict],
    nlp,
    ollama_model: str,
    ollama_host: str,
    sleep: float,
) -> dict:
    predictions = []
    ground_truths = []

    for index, sample in enumerate(rows, start=1):
        content = sample.get("content", "")
        annotations = sample.get("annotation", [])
        if not content or not annotations:
            continue

        gold = annotations_to_entity(annotations)

        llm_pred = None
        ner_pred = None

        if mode in {"llm", "hybrid"}:
            llm_pred = extract_entities(
                content,
                ollama_model=ollama_model,
                ollama_host=ollama_host,
            )
            if llm_pred is None:
                llm_pred = ResumeEntity()
            llm_pred = regex_enrich(llm_pred, content)

        if mode in {"ner", "hybrid"}:
            ner_pred = ner_extract(nlp, content)
            ner_pred = regex_enrich(ner_pred, content)

        if mode == "llm":
            pred = llm_pred or ResumeEntity()
        elif mode == "ner":
            pred = ner_pred or ResumeEntity()
        else:
            pred = merge_entities(llm_pred or ResumeEntity(), ner_pred or ResumeEntity(), content)

        predictions.append(pred)
        ground_truths.append(gold)

        if index % 10 == 0 or index == len(rows):
            print(f"[{mode}] Progress: {index}/{len(rows)}")
        if sleep > 0:
            time.sleep(sleep)

    metrics = evaluate_batch(predictions, ground_truths)
    return metrics


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model_path = Path(args.spacy_model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"spaCy model not found: {model_path}. Train or supply a spaCy NER model first."
        )

    nlp = spacy.load(model_path)

    rows = load_ground_truth(str(dataset_path))
    rows = rows[: max(1, min(args.samples, len(rows)))]

    if args.mode == "all":
        modes = ["llm", "ner", "hybrid"]
    else:
        modes = [args.mode]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {}

    for mode in modes:
        print(f"Starting mode: {mode}")
        metrics = evaluate_mode(
            mode=mode,
            rows=rows,
            nlp=nlp,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            sleep=args.sleep,
        )

        df = results_to_dataframe(metrics)
        csv_path = output_dir / f"comparison_{mode}_{stamp}.csv"
        json_path = output_dir / f"comparison_{mode}_{stamp}.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        macro = metrics.get("macro_avg", {})
        print(
            f"[{mode}] Macro P={macro.get('precision', 0.0):.4f} "
            f"R={macro.get('recall', 0.0):.4f} F1={macro.get('f1', 0.0):.4f}"
        )
        print(f"[{mode}] Saved CSV: {csv_path}")
        print(f"[{mode}] Saved JSON: {json_path}")

        report[mode] = {
            "macro_precision": macro.get("precision", 0.0),
            "macro_recall": macro.get("recall", 0.0),
            "macro_f1": macro.get("f1", 0.0),
            "csv": str(csv_path),
            "json": str(json_path),
        }

    best_mode = max(report.items(), key=lambda kv: kv[1]["macro_f1"])[0]
    print(f"Best mode by macro F1: {best_mode}")


if __name__ == "__main__":
    main()
