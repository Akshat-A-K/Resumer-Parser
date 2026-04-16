"""API-based fine-tuning utilities (OpenAI).

Usage examples:
  python api_finetune.py prepare --max-samples 200
  python api_finetune.py start --api-key YOUR_KEY --base-model gpt-4o-mini
  python api_finetune.py status --api-key YOUR_KEY --job-id ftjob_123
"""

import argparse
import json
import os
from pathlib import Path

import requests

SPLIT_TRAIN = Path("splits/train.jsonl")
OUT_DIR = Path("api_tuning")
TRAIN_FILE = OUT_DIR / "openai_train.jsonl"
LAST_JOB_FILE = OUT_DIR / "last_openai_job.json"

CORE_STRING = ["name", "email", "location", "graduation_year", "years_of_experience"]
CORE_LIST = ["designation", "companies_worked_at", "skills", "college_name", "degree"]

LABEL_TO_FIELD = {
    "NAME": ("name", "string"),
    "EMAIL": ("email", "string"),
    "LOCATION": ("location", "string"),
    "DESIGNATION": ("designation", "list"),
    "COMPANY": ("companies_worked_at", "list"),
    "SKILL": ("skills", "list"),
    "COLLEGE": ("college_name", "list"),
    "DEGREE": ("degree", "list"),
    "GRAD_YEAR": ("graduation_year", "string"),
    "EXPERIENCE": ("years_of_experience", "string"),
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_empty_payload() -> dict:
    payload = {}
    for key in CORE_STRING:
        payload[key] = None
    for key in CORE_LIST:
        payload[key] = []
    return payload


def spans_to_payload(text: str, entities: list[list]) -> dict:
    payload = make_empty_payload()

    for start, end, label in entities:
        field_info = LABEL_TO_FIELD.get(str(label))
        if not field_info:
            continue

        field, kind = field_info
        s = int(start)
        e = int(end)
        if s < 0 or e > len(text) or s >= e:
            continue

        value = text[s:e].strip()
        if not value:
            continue

        if kind == "string":
            if payload[field] is None:
                payload[field] = value
        else:
            if value not in payload[field]:
                payload[field].append(value)

    return payload


def build_prompt(text: str) -> str:
    schema = {
        "name": "string or null",
        "email": "string or null",
        "location": "string or null",
        "designation": ["list of strings"],
        "companies_worked_at": ["list of strings"],
        "skills": ["list of strings"],
        "college_name": ["list of strings"],
        "degree": ["list of strings"],
        "graduation_year": "string or null",
        "years_of_experience": "string or null",
    }
    return (
        "Extract resume fields and return only JSON with this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\nResume text:\n---\n"
        + text[:9000]
        + "\n---"
    )


def prepare_dataset(max_samples: int) -> None:
    if not SPLIT_TRAIN.exists():
        raise FileNotFoundError("splits/train.jsonl not found. Run finetune_prepare.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(SPLIT_TRAIN)
    if max_samples > 0:
        rows = rows[:max_samples]

    with TRAIN_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            text = str(row.get("text", ""))
            entities = row.get("entities", [])
            payload = spans_to_payload(text, entities)

            item = {
                "messages": [
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": build_prompt(text)},
                    {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Prepared OpenAI fine-tuning file: {TRAIN_FILE}")
    print(f"Samples: {len(rows)}")


def openai_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def start_openai_job(api_key: str, base_model: str) -> None:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError("Training file not found. Run prepare first.")

    with TRAIN_FILE.open("rb") as f:
        upload = requests.post(
            "https://api.openai.com/v1/files",
            headers=openai_headers(api_key),
            data={"purpose": "fine-tune"},
            files={"file": (TRAIN_FILE.name, f, "application/jsonl")},
            timeout=300,
        )
    upload.raise_for_status()
    file_payload = upload.json()
    file_id = file_payload["id"]

    create = requests.post(
        "https://api.openai.com/v1/fine_tuning/jobs",
        headers={**openai_headers(api_key), "Content-Type": "application/json"},
        json={
            "training_file": file_id,
            "model": base_model,
        },
        timeout=120,
    )
    create.raise_for_status()
    job_payload = create.json()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LAST_JOB_FILE.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")

    print("Fine-tune job started")
    print(f"Job ID: {job_payload.get('id')}")
    print(f"Status: {job_payload.get('status')}")


def status_openai_job(api_key: str, job_id: str) -> None:
    response = requests.get(
        f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}",
        headers=openai_headers(api_key),
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--max-samples", type=int, default=250)

    start = sub.add_parser("start")
    start.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    start.add_argument("--base-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    status = sub.add_parser("status")
    status.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    status.add_argument("--job-id", required=True)

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_dataset(args.max_samples)
        return

    if args.command == "start":
        if not args.api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        start_openai_job(args.api_key, args.base_model)
        return

    if args.command == "status":
        if not args.api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        status_openai_job(args.api_key, args.job_id)
        return


if __name__ == "__main__":
    main()
