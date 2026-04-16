"""Evaluate the fine-tuned spaCy NER model on test split.

Usage:
    python finetune_evaluate.py
"""

import json
from pathlib import Path

import spacy

MODEL_DIR = Path("models/resume_ner")
TEST_PATH = Path("splits/test.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(nlp, rows: list[dict]) -> tuple[float, float, float]:
    tp = 0
    fp = 0
    fn = 0

    for row in rows:
        text = row.get("text", "")
        gold = {(int(s), int(e), str(l)) for s, e, l in row.get("entities", [])}
        pred_doc = nlp(text)
        pred = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}

        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Model not found. Run finetune_train.py first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError("splits/test.jsonl not found. Run finetune_prepare.py first.")

    nlp = spacy.load(MODEL_DIR)
    test_rows = read_jsonl(TEST_PATH)
    p, r, f1 = evaluate(nlp, test_rows)

    print("Test metrics")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")


if __name__ == "__main__":
    main()
