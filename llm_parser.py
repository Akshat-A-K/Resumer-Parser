"""LLM helpers for resume entity extraction."""

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv

from schema import FIELD_SPECS, LIST_FIELDS, STRING_FIELDS, ResumeEntity, default_selected_fields

load_dotenv()


def _normalize_selected_fields(selected_fields: Optional[list[str]]) -> list[str]:
    if not selected_fields:
        return default_selected_fields()
    allowed = []
    for field in selected_fields:
        if field in FIELD_SPECS and field not in allowed:
            allowed.append(field)
    return allowed or default_selected_fields()


def _build_schema_block(selected_fields: list[str]) -> str:
    lines = ["{"]
    for field in selected_fields:
        if FIELD_SPECS[field]["kind"] == "list":
            lines.append(f'  "{field}": ["list of strings"],')
        else:
            lines.append(f'  "{field}": "string or null",')
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def _build_prompt(resume_text: str, selected_fields: list[str]) -> str:
    schema_block = _build_schema_block(selected_fields)
    return (
        "Extract the requested fields from the resume text.\\n"
        "Return ONLY one valid JSON object.\\n"
        "Rules:\\n"
        "1. Keep keys exactly same as schema.\\n"
        "2. If missing value: null for string fields, [] for list fields.\\n"
        "3. Do not add extra keys.\\n\\n"
        f"Schema to return:\\n{schema_block}\\n\\n"
        f"Resume text:\\n---\\n{resume_text[:12000]}\\n---"
    )


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _clean_payload(data: dict, selected_fields: list[str]) -> dict:
    out = {}

    for field in LIST_FIELDS:
        if field not in selected_fields:
            out[field] = []
            continue
        value = data.get(field)
        if value is None:
            out[field] = []
        elif isinstance(value, str):
            out[field] = [x.strip() for x in value.split(",") if x.strip()]
        elif isinstance(value, list):
            out[field] = [str(x).strip() for x in value if str(x).strip()]
        else:
            out[field] = []

    for field in STRING_FIELDS:
        if field not in selected_fields:
            out[field] = None
            continue
        value = data.get(field)
        if value is None:
            out[field] = None
        elif isinstance(value, list):
            out[field] = str(value[0]).strip() if value else None
        else:
            text = str(value).strip()
            out[field] = text if text else None

    return out


def extract_with_ollama(
    resume_text: str,
    model_name: str,
    host: str,
    selected_fields: list[str],
) -> Optional[ResumeEntity]:
    try:
        import requests

        prompt = _build_prompt(resume_text, selected_fields)
        endpoint = host.rstrip("/") + "/api/chat"
        response = requests.post(
            endpoint,
            json={
                "model": model_name,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()

        payload = response.json()
        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        text = str(message.get("content", "")).strip()

        parsed = _extract_json_object(text)
        if not parsed:
            return None
        return ResumeEntity(**_clean_payload(parsed, selected_fields))
    except Exception:
        return None


def extract_entities(
    resume_text: str,
    provider: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    selected_fields: Optional[list[str]] = None,
    finetuned: bool = False,
    extracted_links: Optional[list[str]] = None,
    use_cache: bool = False,
    truncation_limit: Optional[int] = None,
) -> Optional[ResumeEntity]:
    """Extract entities with the local Ollama model.

    Extra parameters are accepted for compatibility with other branches but are
    intentionally ignored to preserve the parsing approach in this branch.
    """
    _ = provider, gemini_api_key, gemini_model, finetuned, extracted_links, use_cache, truncation_limit

    ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    selected = _normalize_selected_fields(selected_fields)

    if not ollama_model:
        return None
    return extract_with_ollama(resume_text, ollama_model, ollama_host, selected)
