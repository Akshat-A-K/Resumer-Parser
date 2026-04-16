"""Prepare resume NER dataset with 80/10/10 split.

Usage:
    python finetune_prepare.py
"""

import csv
import json
import random
import string
from pathlib import Path

SEED = 42
SOURCE_NER = Path("Entity Recognition in Resumes.json")
SOURCE_CSV = Path("Resume/Resume.csv")
OUT_DIR = Path("splits")

LABEL_MAP = {
    "Name": "NAME",
    "Email Address": "EMAIL",
    "Location": "LOCATION",
    "Designation": "DESIGNATION",
    "Companies worked at": "COMPANY",
    "Skills": "SKILL",
    "College Name": "COLLEGE",
    "Degree": "DEGREE",
    "Graduation Year": "GRAD_YEAR",
    "Years of Experience": "EXPERIENCE",
}


def load_ner_records(path: Path) -> list[dict]:
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


def to_training_sample(record: dict) -> dict | None:
    text = str(record.get("content", ""))
    annotations = record.get("annotation", [])
    if not text:
        return None

    entities = []
    for ann in annotations:
        labels = ann.get("label", [])
        points = ann.get("points", [])
        if not labels or not points:
            continue

        label = labels[0]
        norm_label = LABEL_MAP.get(label)
        if norm_label is None:
            continue

        p = points[0]
        start = p.get("start")
        end = p.get("end")
        if start is None or end is None:
            continue

        # dataset end index is inclusive, convert to exclusive
        start = int(start)
        end = int(end) + 1

        if start < 0 or end > len(text) or start >= end:
            continue

        entities.append([start, end, norm_label])

    entities = sanitize_entities(text, entities)
    entities = remove_overlaps(entities)

    return {"text": text, "entities": entities}


def sanitize_entities(text: str, entities: list[list]) -> list[list]:
    """Clean noisy spans so spaCy training is more stable."""
    punct = set(string.punctuation) | {"•", "●", "▪", "◦", "➢", "❖", "☑", "〓"}
    cleaned: list[list] = []
    seen = set()

    for ent in entities:
        s, e, label = int(ent[0]), int(ent[1]), str(ent[2])
        s = max(0, min(s, len(text)))
        e = max(0, min(e, len(text)))
        if s >= e:
            continue

        while s < e and text[s].isspace():
            s += 1
        while s < e and text[e - 1].isspace():
            e -= 1

        while s < e and text[s] in punct:
            s += 1
        while s < e and text[e - 1] in punct:
            e -= 1

        if s >= e:
            continue

        key = (s, e, label)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append([s, e, label])

    return cleaned


def remove_overlaps(entities: list[list]) -> list[list]:
    """Keep a non-overlapping set of spans (prefer longer spans, then earlier spans)."""
    # Sort by span length desc, then by start asc so longer matches are preferred.
    ranked = sorted(entities, key=lambda x: (-(int(x[1]) - int(x[0])), int(x[0]), int(x[1])))
    kept: list[list] = []

    def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return max(a_start, b_start) < min(a_end, b_end)

    for ent in ranked:
        s, e, _ = int(ent[0]), int(ent[1]), ent[2]
        conflict = False
        for k in kept:
            ks, ke = int(k[0]), int(k[1])
            if overlaps(s, e, ks, ke):
                conflict = True
                break
        if not conflict:
            kept.append(ent)

    kept.sort(key=lambda x: (int(x[0]), int(x[1])))
    return kept


def split_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    random.Random(SEED).shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_resume_csv_text(path: Path, out_path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f_in, out_path.open("w", encoding="utf-8") as f_out:
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
            if not text:
                continue
            f_out.write(text.replace("\n", " ") + "\n")
            count += 1
    return count


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_ner_records(SOURCE_NER)
    samples = []
    for rec in raw:
        item = to_training_sample(rec)
        if item is not None:
            samples.append(item)

    train, val, test = split_records(samples)

    save_jsonl(OUT_DIR / "train.jsonl", train)
    save_jsonl(OUT_DIR / "val.jsonl", val)
    save_jsonl(OUT_DIR / "test.jsonl", test)

    resume_count = export_resume_csv_text(SOURCE_CSV, OUT_DIR / "resume_unlabeled.txt")

    print(f"Total labeled samples: {len(samples)}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Resume.csv unlabeled texts exported: {resume_count}")
    print("Saved in ./splits")


if __name__ == "__main__":
    main()
