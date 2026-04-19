# prepare resume datasets for llm finetuning with 80/10/10 split

import csv
import json
import random
from pathlib import Path

from pypdf import PdfReader
from evaluator import annotations_to_dict, LABEL_MAP

SEED = 42
SOURCE_NER = Path("Entity Recognition in Resumes.json")
SOURCE_CSV = Path("Resume/Resume.csv")
SOURCE_PDF_DIR = Path("data/data")
OUT_DIR = Path("splits")


def load_ner_records(path: Path) -> list[dict]:
    # load newline delimited json file
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def record_to_training_pair(record: dict) -> dict | None:
    # convert ner annotated record to text expected_output pair
    text = str(record.get("content", "")).strip()
    annotations = record.get("annotation", [])
    if not text or not annotations:
        return None

    expected = annotations_to_dict(annotations)
    if not expected:
        return None

    return {"text": text, "expected_output": expected}


def compute_quality_score(pair: dict) -> float:
    # score training pair by richness of annotations for fewshot selection
    expected = pair.get("expected_output", {})
    text = pair.get("text", "")

    field_count = 0
    item_count = 0
    for k, v in expected.items():
        if v is None or v == []:
            continue
        field_count += 1
        if isinstance(v, list):
            item_count += len(v)
        else:
            item_count += 1

    text_len = len(text)
    length_bonus = min(text_len / 2000, 1.0)
    return field_count * 3 + item_count + length_bonus


def split_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    # 80/10/10 split with seed 42
    rng = random.Random(SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_fewshot_examples(train_set: list[dict], count: int = 10) -> list[dict]:
    # select best training examples for fewshot prompting
    scored = [(compute_quality_score(ex), ex) for ex in train_set]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for _, ex in scored:
        if len(selected) >= count:
            break
        selected.append(ex)

    return selected


def export_resume_csv_text(path: Path, out_path: Path) -> int:
    # export text from resume csv
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        candidates = ["Resume_str", "Resume", "resume", "text", "Text"]

        for row in reader:
            text_col = None
            for c in candidates:
                if c in row:
                    text_col = c
                    break
            if text_col is None:
                continue
            text = (row.get(text_col) or "").strip()
            if not text or len(text) < 100:
                continue
            f_out.write(text.replace("\n", " ") + "\n")
            count += 1
    return count


def export_pdf_corpus_text(pdf_root: Path, out_path: Path, max_files: int = 2500) -> int:
    # export plain text from pdf resumes
    if not pdf_root.exists():
        return 0

    pdf_paths = sorted(pdf_root.rglob("*.pdf"))[:max_files]
    count = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for pdf in pdf_paths:
            try:
                reader = PdfReader(str(pdf))
                parts = []
                for page in reader.pages:
                    parts.append((page.extract_text() or "").strip())
                text = " ".join([p for p in parts if p]).strip()
                if not text or len(text) < 100:
                    continue
                f_out.write(text.replace("\n", " ") + "\n")
                count += 1
            except Exception:
                continue

    return count


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading NER data from {SOURCE_NER}...")
    raw = load_ner_records(SOURCE_NER)
    print(f"  Loaded {len(raw)} raw records")

    samples = []
    for rec in raw:
        pair = record_to_training_pair(rec)
        if pair is not None:
            samples.append(pair)
    print(f"  Converted {len(samples)} valid training pairs")

    train, val, test = split_records(samples)

    save_jsonl(OUT_DIR / "train.jsonl", train)
    save_jsonl(OUT_DIR / "val.jsonl", val)
    save_jsonl(OUT_DIR / "test.jsonl", test)

    fewshot = select_fewshot_examples(train, count=10)
    with (OUT_DIR / "fewshot_examples.json").open("w", encoding="utf-8") as f:
        json.dump(fewshot, f, indent=2, ensure_ascii=False)
    print(f"  Selected {len(fewshot)} fewshot examples")

    resume_count = export_resume_csv_text(SOURCE_CSV, OUT_DIR / "corpus_unlabeled.txt")
    pdf_count = export_pdf_corpus_text(SOURCE_PDF_DIR, OUT_DIR / "corpus_pdf.txt")

    stats = {
        "seed": SEED,
        "split_ratio": "80/10/10",
        "total_labeled_samples": len(samples),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "fewshot_count": len(fewshot),
        "resume_csv_unlabeled": resume_count,
        "pdf_corpus_unlabeled": pdf_count,
    }
    with (OUT_DIR / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTotal labeled samples: {len(samples)}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Fewshot examples: {len(fewshot)}")
    print(f"Resume.csv unlabeled: {resume_count}")
    print(f"PDF corpus unlabeled: {pdf_count}")
    print(f"Saved in ./{OUT_DIR}")


if __name__ == "__main__":
    main()
