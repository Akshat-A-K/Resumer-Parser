"""
evaluator.py - Comprehensive evaluation metrics for resume entity extraction.

Metrics
-------
Scalar fields  (name, email, location, graduation_year, years_of_experience):
  • Exact Match (EM)         – binary, after normalisation
  • Fuzzy Match              – SequenceMatcher ratio  (0-1)
  • Token-level Precision / Recall / F1

List fields  (designation, companies, skills, college, degree):
  • Set-based Precision / Recall / F1  with fuzzy matching
  • Jaccard Similarity

Aggregate:
  • Per-entity-type micro metrics
  • Macro-averaged Precision / Recall / F1
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd

from schema import ResumeEntity

# ── Constants ──────────────────────────────────────────────────────────

LABEL_MAP = {
    "Name": "name",
    "Email Address": "email",
    "Location": "location",
    "Designation": "designation",
    "Companies worked at": "companies_worked_at",
    "Skills": "skills",
    "College Name": "college_name",
    "Degree": "degree",
    "Graduation Year": "graduation_year",
    "Years of Experience": "years_of_experience",
}

SINGLE_FIELDS = {"name", "email", "location", "graduation_year", "years_of_experience"}
LIST_FIELDS = {"designation", "companies_worked_at", "skills", "college_name", "degree"}

# ── Ground-truth helpers ──────────────────────────────────────────────


def load_ground_truth(json_path: str) -> List[dict]:
    """
    Load the annotated ground-truth JSON file.

    The file is newline-delimited JSON (one object per line).
    """
    data = []
    with open(json_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def annotations_to_entity(annotations: List[dict]) -> ResumeEntity:
    """Convert raw annotation list to a ResumeEntity ground-truth object."""
    entity: Dict = {
        "name": None,
        "email": None,
        "location": None,
        "graduation_year": None,
        "years_of_experience": None,
        "designation": [],
        "companies_worked_at": [],
        "skills": [],
        "college_name": [],
        "degree": [],
    }

    for ann in annotations:
        raw_label = ann.get("label")
        if isinstance(raw_label, list):
            if not raw_label:
                continue
            label = str(raw_label[0]).strip()
        elif isinstance(raw_label, str):
            label = raw_label.strip()
        else:
            continue

        field = LABEL_MAP.get(label)
        if field is None:
            continue

        points = ann.get("points")
        if not isinstance(points, list) or not points:
            continue

        text = str(points[0].get("text", "")).strip()
        if not text:
            continue

        if field in SINGLE_FIELDS:
            if entity[field] is None:
                entity[field] = text
        else:
            # Deduplicate ground-truth list entries
            if text not in entity[field]:
                entity[field].append(text)

    return ResumeEntity(**entity)


# ── Text normalisation ────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ── Scalar-field metrics ──────────────────────────────────────────────


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalised strings are identical, else 0.0."""
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def fuzzy_score(pred: str, gold: str) -> float:
    """SequenceMatcher ratio on normalised strings (0-1)."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return SequenceMatcher(None, normalize(pred), normalize(gold)).ratio()


def token_prf(pred: str, gold: str) -> Tuple[float, float, float]:
    """
    Token-level precision, recall, F1.

    Each string is treated as a bag of words; overlap is counted.
    """
    p_tok = set(normalize(pred).split())
    g_tok = set(normalize(gold).split())

    if not p_tok and not g_tok:
        return 1.0, 1.0, 1.0
    if not p_tok:
        return 0.0, 0.0, 0.0
    if not g_tok:
        return 0.0, 0.0, 0.0

    common = p_tok & g_tok
    prec = len(common) / len(p_tok)
    rec = len(common) / len(g_tok)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ── List-field metrics ────────────────────────────────────────────────


def jaccard(pred_set: set, gold_set: set) -> float:
    """|A ∩ B| / |A ∪ B|."""
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    return len(pred_set & gold_set) / len(pred_set | gold_set)


def _fuzzy_raw(a: str, b: str) -> float:
    """Sequence similarity on already-normalised strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def set_prf(
    pred_list: List[str],
    gold_list: List[str],
    threshold: float = 0.75,
) -> Tuple[float, float, float]:
    """
    Set-based precision / recall / F1 with fuzzy matching.

    A predicted item counts as a match if its best fuzzy score against
    any unmatched gold item exceeds *threshold*.
    """
    if not pred_list and not gold_list:
        return 1.0, 1.0, 1.0
    if not pred_list:
        return 0.0, 0.0, 0.0
    if not gold_list:
        return 0.0, 0.0, 0.0

    pred_n = [normalize(p) for p in pred_list]
    gold_n = [normalize(g) for g in gold_list]

    matched_pred = 0
    matched_gold: set = set()

    for p in pred_n:
        best_score = 0.0
        best_idx = -1
        for i, g in enumerate(gold_n):
            if i in matched_gold:
                continue
            s = _fuzzy_raw(p, g)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_score >= threshold and best_idx >= 0:
            matched_pred += 1
            matched_gold.add(best_idx)

    prec = matched_pred / len(pred_n)
    rec = len(matched_gold) / len(gold_n)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ── Single-sample evaluation ─────────────────────────────────────────


def evaluate_single(pred: ResumeEntity, gold: ResumeEntity) -> Dict[str, dict]:
    """Return per-field metrics for one (prediction, ground-truth) pair."""
    results: Dict[str, dict] = {}

    for field in SINGLE_FIELDS:
        pv = getattr(pred, field) or ""
        gv = getattr(gold, field) or ""
        em = exact_match(pv, gv)
        fs = fuzzy_score(pv, gv)
        p, r, f1 = token_prf(pv, gv)
        results[field] = dict(
            exact_match=em, fuzzy_match=fs,
            precision=p, recall=r, f1=f1,
            has_gold=bool(gv), has_pred=bool(pv),
        )

    for field in LIST_FIELDS:
        pl = getattr(pred, field) or []
        gl = getattr(gold, field) or []
        p, r, f1 = set_prf(pl, gl)
        ps = {normalize(x) for x in pl}
        gs = {normalize(x) for x in gl}
        jac = jaccard(ps, gs)
        results[field] = dict(
            precision=p, recall=r, f1=f1, jaccard=jac,
            num_predicted=len(pl), num_gold=len(gl),
            has_gold=bool(gl), has_pred=bool(pl),
        )

    return results


# ── Batch evaluation & aggregation ────────────────────────────────────


def evaluate_batch(
    predictions: List[ResumeEntity],
    ground_truths: List[ResumeEntity],
) -> Dict:
    """
    Evaluate a batch and return aggregate metrics.

    Returns
    -------
    dict with keys:
      per_field   – per-entity-type averaged metrics
      macro_avg   – macro-averaged P / R / F1
      num_samples – number of samples evaluated
    """
    all_results = [
        evaluate_single(p, g) for p, g in zip(predictions, ground_truths)
    ]

    all_fields = list(SINGLE_FIELDS) + list(LIST_FIELDS)
    field_metrics: Dict[str, dict] = {}

    for field in all_fields:
        field_data = [r[field] for r in all_results if field in r]
        if not field_data:
            continue

        with_gold = [d for d in field_data if d.get("has_gold", False)]
        support = len(with_gold)

        avg: Dict = {}
        for metric in ("precision", "recall", "f1"):
            vals = [d[metric] for d in field_data if metric in d]
            avg[metric] = sum(vals) / len(vals) if vals else 0.0

        if field in SINGLE_FIELDS:
            for extra in ("exact_match", "fuzzy_match"):
                vals = [d[extra] for d in field_data if extra in d]
                avg[extra] = sum(vals) / len(vals) if vals else 0.0

        if field in LIST_FIELDS:
            vals = [d["jaccard"] for d in field_data if "jaccard" in d]
            avg["jaccard"] = sum(vals) / len(vals) if vals else 0.0

        avg["support"] = support
        avg["total_samples"] = len(field_data)
        field_metrics[field] = avg

    # Macro average
    ps = [field_metrics[f]["precision"] for f in all_fields if f in field_metrics]
    rs = [field_metrics[f]["recall"] for f in all_fields if f in field_metrics]
    fs = [field_metrics[f]["f1"] for f in all_fields if f in field_metrics]

    macro = dict(
        precision=sum(ps) / len(ps) if ps else 0.0,
        recall=sum(rs) / len(rs) if rs else 0.0,
        f1=sum(fs) / len(fs) if fs else 0.0,
    )

    return dict(per_field=field_metrics, macro_avg=macro, num_samples=len(all_results))


# ── DataFrame helper ──────────────────────────────────────────────────


def results_to_dataframe(eval_results: Dict) -> pd.DataFrame:
    """Convert evaluation dict to a tidy DataFrame for display / export."""
    rows = []
    for field, m in eval_results["per_field"].items():
        row = {"Entity Type": field.replace("_", " ").title()}
        row["Precision"] = round(m["precision"], 4)
        row["Recall"] = round(m["recall"], 4)
        row["F1 Score"] = round(m["f1"], 4)
        if "exact_match" in m:
            row["Exact Match"] = round(m["exact_match"], 4)
        if "fuzzy_match" in m:
            row["Fuzzy Match"] = round(m["fuzzy_match"], 4)
        if "jaccard" in m:
            row["Jaccard"] = round(m["jaccard"], 4)
        row["Support"] = m["support"]
        rows.append(row)

    mac = eval_results["macro_avg"]
    rows.append(
        {
            "Entity Type": "MACRO AVG",
            "Precision": round(mac["precision"], 4),
            "Recall": round(mac["recall"], 4),
            "F1 Score": round(mac["f1"], 4),
            "Support": eval_results["num_samples"],
        }
    )
    return pd.DataFrame(rows)
