"""Fine-tune a spaCy NER model on prepared splits.

Best default choice in this script:
- Try transformer model en_core_web_trf (best quality)
- Fallback to en_core_web_sm if transformer is unavailable
- Final fallback to blank English model

Usage:
    python finetune_train.py
"""

import json
import os
import random
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
from spacy.util import minibatch

SEED = 42
SPLIT_DIR = Path("splits")
MODEL_DIR = Path("models/resume_ner")
EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "15"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_base_model() -> tuple[spacy.language.Language, str]:
    try:
        print("Using en_core_web_trf as base model")
        return spacy.load("en_core_web_trf"), "trf"
    except Exception:
        try:
            print("en_core_web_trf not found. Using en_core_web_sm")
            return spacy.load("en_core_web_sm"), "sm"
        except Exception:
            print("No pre-trained spaCy model found. Using blank English model")
            return spacy.blank("en"), "blank"


def make_examples(nlp, rows: list[dict]) -> list[Example]:
    examples = []
    skipped = 0
    for row in rows:
        text = row.get("text", "")
        entities = row.get("entities", [])
        if not text:
            continue

        doc = nlp.make_doc(text)
        biluo = offsets_to_biluo_tags(doc, entities)
        if "-" in biluo:
            skipped += 1
            continue

        try:
            examples.append(Example.from_dict(doc, {"entities": entities}))
        except Exception:
            skipped += 1
            continue

    if skipped:
        print(f"Skipped {skipped} invalid/misaligned samples")
    return examples


def evaluate_simple(nlp, rows: list[dict]) -> tuple[float, float, float]:
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
    random.seed(SEED)

    if not (SPLIT_DIR / "train.jsonl").exists() or not (SPLIT_DIR / "val.jsonl").exists():
        raise FileNotFoundError("Missing splits. Run finetune_prepare.py first.")

    train_rows = read_jsonl(SPLIT_DIR / "train.jsonl")
    val_rows = read_jsonl(SPLIT_DIR / "val.jsonl")
    if not train_rows:
        raise ValueError("No training rows found in splits/train.jsonl")

    nlp, source = load_base_model()

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for row in train_rows:
        for _, _, label in row.get("entities", []):
            ner.add_label(label)

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other_pipes):
        # Pretrained models should continue training with resume_training.
        # initialize() is only needed when training from a blank model.
        if source == "blank":
            optimizer = nlp.initialize(get_examples=lambda: make_examples(nlp, train_rows))
        else:
            optimizer = nlp.resume_training()

        best_f1 = -1.0
        for epoch in range(EPOCHS):
            random.shuffle(train_rows)
            losses = {}

            for batch_rows in minibatch(train_rows, size=8):
                examples = make_examples(nlp, batch_rows)
                nlp.update(examples, sgd=optimizer, losses=losses, drop=0.2)

            p, r, f1 = evaluate_simple(nlp, val_rows)
            print(
                f"Epoch {epoch + 1:02d} | loss={losses.get('ner', 0):.4f} "
                f"| val_p={p:.4f} val_r={r:.4f} val_f1={f1:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                nlp.to_disk(MODEL_DIR)
                print(f"Saved best model at epoch {epoch + 1:02d} (val_f1={f1:.4f})")

    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
