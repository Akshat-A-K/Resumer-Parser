# evaluation metrics for resume entity extraction

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import pandas as pd
from schema import LIST_FIELDS as SCHEMA_LIST_FIELDS
from schema import STRING_FIELDS as SCHEMA_STRING_FIELDS
from schema import ResumeEntity

# map annotation labels to schema field names
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

SINGLE_FIELDS = set(SCHEMA_STRING_FIELDS)
LIST_FIELDS_SET = set(SCHEMA_LIST_FIELDS)


def load_ground_truth(json_path: str) -> List[dict]:
    # load newline delimited json annotation file
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
    # convert raw annotation list to resume entity
    entity: Dict = {}
    for field in SCHEMA_STRING_FIELDS:
        entity[field] = None
    for field in SCHEMA_LIST_FIELDS:
        entity[field] = []

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
        elif field in LIST_FIELDS_SET:
            if text not in entity[field]:
                entity[field].append(text)

    return ResumeEntity(**entity)


def annotations_to_dict(annotations: List[dict]) -> dict:
    # convert raw annotations to plain dict for training data
    entity: Dict = {}
    for field in SCHEMA_STRING_FIELDS:
        entity[field] = None
    for field in SCHEMA_LIST_FIELDS:
        entity[field] = []

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
        elif field in LIST_FIELDS_SET:
            if text not in entity[field]:
                entity[field].append(text)

    cleaned = {}
    for k, v in entity.items():
        if v is not None and v != []:
            cleaned[k] = v
    return cleaned


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _normalize_numeric_string(value: str) -> str:
    m = re.fullmatch(r"\s*(\d+)\.0+\s*", value)
    if m:
        return m.group(1)
    return value.strip()


def _normalize_date_string(value: str) -> str:
    text = value.strip()
    if not text:
        return text

    if re.fullmatch(r"\d{4}", text):
        return text

    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", text)
    if m:
        year, month = m.group(1), m.group(2).zfill(2)
        return f"{year}-{month}"

    m = re.fullmatch(r"(\d{1,2})[-/](\d{4})", text)
    if m:
        month, year = m.group(1).zfill(2), m.group(2)
        return f"{year}-{month}"

    month_map = {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02",
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05",
        "jun": "06", "june": "06",
        "jul": "07", "july": "07",
        "aug": "08", "august": "08",
        "sep": "09", "sept": "09", "september": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12",
    }

    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", text)
    if m:
        month = month_map.get(m.group(1).lower())
        if month:
            return f"{m.group(2)}-{month}"

    m = re.fullmatch(r"(\d{4})\s+([A-Za-z]+)", text)
    if m:
        month = month_map.get(m.group(2).lower())
        if month:
            return f"{m.group(1)}-{month}"

    return text


def _normalize_for_eval(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip()
    text = _normalize_numeric_string(text)
    if "date" in text or re.search(r"\b\d{4}\b", text):
        text = _normalize_date_string(text)
    return normalize(text)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_for_eval(pred) == _normalize_for_eval(gold) else 0.0


def fuzzy_score(pred: str, gold: str) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return SequenceMatcher(None, _normalize_for_eval(pred), _normalize_for_eval(gold)).ratio()


def token_prf(pred: str, gold: str) -> Tuple[float, float, float]:
    # token level precision recall f1
    p_tok = set(_normalize_for_eval(pred).split())
    g_tok = set(_normalize_for_eval(gold).split())

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


def jaccard(pred_set: set, gold_set: set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    return len(pred_set & gold_set) / len(pred_set | gold_set)


def _fuzzy_raw(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def set_prf(pred_list, gold_list, threshold=0.75):
    # set based precision recall f1 with fuzzy matching
    if not pred_list and not gold_list:
        return 1.0, 1.0, 1.0
    if not pred_list:
        return 0.0, 0.0, 0.0
    if not gold_list:
        return 0.0, 0.0, 0.0

    pred_n = [_normalize_for_eval(p) for p in pred_list]
    gold_n = [_normalize_for_eval(g) for g in gold_list]

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


def evaluate_single(pred: ResumeEntity, gold: ResumeEntity) -> Dict[str, dict]:
    # per field metrics for one prediction ground truth pair
    results: Dict[str, dict] = {}

    for field in SINGLE_FIELDS:
        pv = getattr(pred, field, None) or ""
        gv = getattr(gold, field, None) or ""
        em = exact_match(pv, gv)
        fs = fuzzy_score(pv, gv)
        p, r, f1 = token_prf(pv, gv)
        results[field] = dict(
            exact_match=em, fuzzy_match=fs,
            precision=p, recall=r, f1=f1,
            has_gold=bool(gv), has_pred=bool(pv),
        )

    for field in LIST_FIELDS_SET:
        pl = getattr(pred, field, None) or []
        gl = getattr(gold, field, None) or []
        p, r, f1 = set_prf(pl, gl)
        ps = {_normalize_for_eval(x) for x in pl}
        gs = {_normalize_for_eval(x) for x in gl}
        jac = jaccard(ps, gs)
        results[field] = dict(
            precision=p, recall=r, f1=f1, jaccard=jac,
            num_predicted=len(pl), num_gold=len(gl),
            has_gold=bool(gl), has_pred=bool(pl),
        )

    return results


def evaluate_batch(predictions, ground_truths) -> Dict:
    # evaluate a batch and return aggregate metrics
    all_results = [
        evaluate_single(p, g) for p, g in zip(predictions, ground_truths)
    ]

    all_fields = list(SINGLE_FIELDS) + list(LIST_FIELDS_SET)
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

        if field in LIST_FIELDS_SET:
            vals = [d["jaccard"] for d in field_data if "jaccard" in d]
            avg["jaccard"] = sum(vals) / len(vals) if vals else 0.0

        avg["support"] = support
        avg["total_samples"] = len(field_data)
        field_metrics[field] = avg

    # macro average only over fields with ground truth
    scored_fields = [f for f in all_fields if f in field_metrics and field_metrics[f].get("support", 0) > 0]
    ps = [field_metrics[f]["precision"] for f in scored_fields]
    rs = [field_metrics[f]["recall"] for f in scored_fields]
    fs = [field_metrics[f]["f1"] for f in scored_fields]

    macro = dict(
        precision=sum(ps) / len(ps) if ps else 0.0,
        recall=sum(rs) / len(rs) if rs else 0.0,
        f1=sum(fs) / len(fs) if fs else 0.0,
    )

    return dict(per_field=field_metrics, macro_avg=macro, num_samples=len(all_results))


def results_to_dataframe(eval_results: Dict) -> pd.DataFrame:
    # convert evaluation dict to dataframe for display
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
