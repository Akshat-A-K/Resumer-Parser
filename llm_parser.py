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


def extract_with_gemini(
    resume_text: str,
    api_key: str,
    selected_fields: list[str],
) -> Optional[ResumeEntity]:
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = _build_prompt(resume_text, selected_fields)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 2048},
        )
        text = getattr(response, "text", "") or ""
        parsed = _extract_json_object(text)
        if not parsed:
            return None
        return ResumeEntity(**_clean_payload(parsed, selected_fields))
    except Exception:
        return None


def extract_with_groq(
    resume_text: str,
    api_key: str,
    selected_fields: list[str],
) -> Optional[ResumeEntity]:
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        prompt = _build_prompt(resume_text, selected_fields)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        text = completion.choices[0].message.content or ""
        parsed = _extract_json_object(text)
        if not parsed:
            return None
        return ResumeEntity(**_clean_payload(parsed, selected_fields))
    except Exception:
        return None


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


def extract_with_openai(
    resume_text: str,
    api_key: str,
    model_name: str,
    selected_fields: list[str],
) -> Optional[ResumeEntity]:
    try:
        import requests

        prompt = _build_prompt(resume_text, selected_fields)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()

        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            return None

        text = str(choices[0].get("message", {}).get("content", "")).strip()
        parsed = _extract_json_object(text)
        if not parsed:
            return None
        return ResumeEntity(**_clean_payload(parsed, selected_fields))
    except Exception:
        return None


def extract_entities(
    resume_text: str,
    provider: str = "gemini",
    gemini_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    selected_fields: Optional[list[str]] = None,
) -> Optional[ResumeEntity]:
    provider = (provider or "gemini").strip().lower()
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
    groq_key = groq_key or os.getenv("GROQ_API_KEY")
    ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    openai_key = openai_key or os.getenv("OPENAI_API_KEY")
    openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    selected = _normalize_selected_fields(selected_fields)

    if provider == "gemini":
        if not gemini_key:
            return None
        return extract_with_gemini(resume_text, gemini_key, selected)

    if provider == "groq":
        if not groq_key:
            return None
        return extract_with_groq(resume_text, groq_key, selected)

    if provider == "ollama-local":
        if not ollama_model:
            return None
        return extract_with_ollama(resume_text, ollama_model, ollama_host, selected)

    if provider == "openai":
        if not openai_key:
            return None
        return extract_with_openai(resume_text, openai_key, openai_model, selected)

    return None
