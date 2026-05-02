# LLM based resume entity extraction with ollama and gemini
# Features: structured education/experience/projects, link injection, SHA-256 caching

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from schema import (
    FIELD_SPECS, LIST_FIELDS, STRING_FIELDS, STRUCTURED_FIELDS,
    ResumeEntity, default_selected_fields,
)

load_dotenv()

logger = logging.getLogger(__name__)

FEWSHOT_PATH = Path("splits/fewshot_examples.json")

# stores the last extraction error for UI display
_last_error: str = ""

def get_last_error() -> str:
    return _last_error
CACHE_DIR = Path("cache")

FIELD_DESCRIPTIONS = {
    "name": "Full name",
    "email": "Email address",
    "phone": "Phone number",
    "location": "Current city/country",
    "linkedin": "LinkedIn URL",
    "github": "GitHub URL",
    "portfolio": "Portfolio/Personal site URL",
    "twitter": "Twitter/X URL",
    "other_links": "Any other social/profile links",
    "designation": "Current or past job titles",
    "companies_worked_at": "Names of past employers/organizations",
    "skills": "All technical and soft skills",
    "years_of_experience": "Total years of professional experience",
    "summary": "Short professional bio",
    "college_name": "Institutions attended",
    "degree": "Degrees earned",
    "graduation_year": "Year of graduation",
    "certifications": "Licenses or certifications",
    "projects": "Names of projects",
    "languages": "Languages spoken",
    "achievements": "Awards or honors",
    "publications": "Papers or articles",
    "hobbies": "Interests",
    "references": "Professional references",
    "skills_categorized": "Skills grouped by category (Languages, Frameworks, etc.)",
    "education_details": "Structured list: {institution, degree, field_of_study, cgpa, location, start_year, end_year}",
    "experience_details": "Structured list: {company, role, location, start_date, end_date, description, tech_stack}",
    "projects_detailed": "Structured list: {name, tech_stack, description, link, date}",
}


def _classify_link(url: str) -> str:
    """Classify a URL into a human-readable category."""
    url_lower = url.lower().rstrip("/")
    if "linkedin.com" in url_lower:
        return "LinkedIn Profile"
    if "github.com" in url_lower:
        match = re.search(r'github\.com/([^/?#]+)(?:/([^/?#]+))?', url_lower)
        if match:
            username = match.group(1)
            repo = match.group(2)
            if username in ("features", "marketplace", "explore", "topics", "settings"):
                return "GitHub"
            if repo and repo not in ("", "repositories", "stars", "followers", "following"):
                return "GitHub Project"
            return "GitHub Profile"
        return "GitHub"
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "Twitter/X Profile"
    if "kaggle.com" in url_lower:
        return "Kaggle Profile"
    if "stackoverflow.com" in url_lower or "stackexchange.com" in url_lower:
        return "StackOverflow"
    if "medium.com" in url_lower:
        return "Blog"
    if any(kw in url_lower for kw in ("portfolio", "personal", ".me", ".dev", "about.me")):
        return "Portfolio"
    if "leetcode.com" in url_lower:
        return "LeetCode Profile"
    if "codeforces.com" in url_lower:
        return "Codeforces Profile"
    if "codechef.com" in url_lower:
        return "CodeChef Profile"
    if url_lower.startswith("mailto:"):
        return "Email"
    return "Link"


def _map_project_links(projects: list, extracted_links: Optional[list[str]]) -> list:
    """Map GitHub project URLs to projects_detailed entries when missing."""
    if not projects or not extracted_links:
        return projects

    def _repo_slug(url: str) -> Optional[str]:
        m = re.search(r"github\.com/([^/]+)/([^/?#]+)", url, re.IGNORECASE)
        if not m:
            return None
        return m.group(2)

    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    gh_links = []
    for link in extracted_links:
        if _repo_slug(link):
            gh_links.append(link)

    if not gh_links:
        return projects

    # if only one project and one link, map directly
    if len(projects) == 1 and len(gh_links) == 1:
        proj = projects[0]
        if isinstance(proj, dict) and (not proj.get("link") or str(proj.get("link")).lower() == "github"):
            proj["link"] = gh_links[0]
        return projects

    # map by slug overlap
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        if proj.get("link") and str(proj.get("link")).strip().lower() not in ("github", "link", "null", "none", "n/a"):
            continue
        name = proj.get("name") or ""
        if not name:
            continue
        name_slug = _slug(name)
        name_tokens = set(name_slug.split("-")) - {""}
        for link in gh_links:
            repo = _repo_slug(link)
            if not repo:
                continue
            repo_slug = _slug(repo)
            repo_tokens = set(repo_slug.split("-")) - {""}
            if repo_slug and (repo_slug in name_slug or name_slug in repo_slug or repo_tokens.intersection(name_tokens)):
                proj["link"] = link
                break

    return projects

def _normalize_selected_fields(selected_fields: Optional[list[str]]) -> list[str]:
    if not selected_fields:
        return default_selected_fields()
    allowed = []
    for field in selected_fields:
        if field in FIELD_SPECS and field not in allowed:
            allowed.append(field)
    return allowed or default_selected_fields()


def _build_schema_block(selected_fields: list[str]) -> str:
    """Build annotated JSON schema block for prompt."""
    lines = ["{"]
    for field in selected_fields:
        desc = FIELD_DESCRIPTIONS.get(field, "")
        spec = FIELD_SPECS.get(field, {})
        kind = spec.get("kind", "string")

        if kind == "list":
            lines.append(f'  "{field}": ["..."],  // {desc}')
            lines.append(
                f'  "{field}": {{"CategoryName": ["Skill1", "Skill2"]}},  '
                f'// {desc}'
            )
        elif kind == "structured":
            if field == "education_details":
                lines.append(
                    f'  "{field}": ['
                    '{"institution": "...", "degree": "...", "field_of_study": "...", '
                    '"cgpa": "...", "location": "...", "start_year": "...", "end_year": "..."}],  '
                    f'// {desc}'
                )
            elif field == "experience_details":
                lines.append(
                    f'  "{field}": ['
                    '{"company": "...", "role": "...", "location": "...", '
                    '"start_date": "...", "end_date": "...", "description": "...", '
                    '"tech_stack": ["..."]}],  '
                    f'// {desc}'
                )
            elif field == "projects_detailed":
                lines.append(
                    f'  "{field}": ['
                    '{"name": "...", "tech_stack": ["..."], "description": "...", '
                    '"link": "...", "date": "..."}],  '
                    f'// {desc}'
                )
        else:
            lines.append(f'  "{field}": "...",  // {desc}')

    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def _build_json_template(selected_fields: list[str]) -> str:
    """Build a strict JSON template with all keys present."""
    def _default_for_field(field: str):
        kind = FIELD_SPECS.get(field, {}).get("kind", "string")
        if kind in ("list", "structured"):
            return []
        if kind == "dict":
            return {}
        return None

    template = {field: _default_for_field(field) for field in selected_fields}
    return json.dumps(template, ensure_ascii=False)


def _load_fewshot_examples() -> list[dict]:
    """Load curated fewshot examples from splits directory."""
    if not FEWSHOT_PATH.exists():
        return []
    try:
        with FEWSHOT_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _build_fewshot_block(examples: list[dict], selected_fields: list[str]) -> str:
    """Format fewshot examples for injection into prompt."""
    if not examples:
        return ""
    blocks = []
    for i, ex in enumerate(examples[:5], start=1):
        text_snippet = str(ex.get("text", ""))[:800]
        expected = ex.get("expected_output", {})
        filtered = {}
        for f in selected_fields:
            if f in expected:
                filtered[f] = expected[f]
        blocks.append(
            f"--- Example {i} ---\n"
            f"Resume text:\n{text_snippet}\n\n"
            f"Expected JSON output:\n{json.dumps(filtered, indent=2, ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


SYSTEM_PROMPT = (
    "You are a precise resume parser. Extract information ONLY from the provided text into the requested JSON schema. "
    "Do not fabricate values. Return ONLY valid JSON. No explanations. Use null if not found. For lists, use []. "
    "For structured fields, create a separate object for EACH entry."
)


def _build_prompt(
    resume_text: str,
    selected_fields: list[str],
    fewshot: bool = False,
    extracted_links: Optional[list[str]] = None,
    truncation_limit: int = 8000,
) -> str:
    schema_block = _build_schema_block(selected_fields)
    template = _build_json_template(selected_fields)
    allowed_keys = ", ".join(selected_fields)

    prompt_parts = [
        "Extract the requested fields from the resume text below.\n",
        "IMPORTANT: Return ONLY a single JSON object.\n",
        "IMPORTANT: Use ONLY the keys listed in the schema. Do NOT add any extra keys.\n",
        "If a value is missing, use null for strings and [] for lists.\n",
        f"Allowed keys: {allowed_keys}\n",
        f"JSON template (fill values only, keep keys):\n{template}\n",
        f"Schema (return exactly these keys):\n{schema_block}\n",
    ]

    # inject extracted links with classification so the LLM has clean URLs
    if extracted_links:
        classified = []
        for url in extracted_links:
            label = _classify_link(url)
            classified.append(f"  - [{label}] {url}")
        links_block = "\n".join(classified)
        prompt_parts.append(
            f"Links found in the document (use these for linkedin, github, portfolio, "
            f"twitter, other_links fields  labels indicate the link type):\n{links_block}\n"
        )

    if fewshot:
        examples = _load_fewshot_examples()
        if examples:
            fewshot_block = _build_fewshot_block(examples, selected_fields)
            prompt_parts.append(
                f"Here are examples of correct extractions:\n\n{fewshot_block}\n\n"
                "Now extract from the following resume using the same format:\n"
            )

    prompt_parts.append(f"Resume text:\n---\n{resume_text[:truncation_limit]}\n---")
    return "\n".join(prompt_parts)


def _split_sections(resume_text: str) -> dict:
    """Split resume text into coarse sections by headings."""
    headings = {
        "education": "education",
        "experience": "experience",
        "projects": "projects",
        "technical skills": "skills",
        "skills": "skills",
        "achievements": "achievements",
        "co-curricular activities": "activities",
        "activities": "activities",
    }

    sections = {"contact": []}
    current = "contact"
    for line in resume_text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        key = ln.lower()
        if key in headings:
            current = headings[key]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(ln)

    return {k: "\n".join(v) for k, v in sections.items()}


def _merge_section_results(selected_fields: list[str], parts: list[dict]) -> dict:
    """Merge partial outputs into a full schema payload."""
    merged = {f: None for f in selected_fields}
    for f in selected_fields:
        kind = FIELD_SPECS.get(f, {}).get("kind", "string")
        if kind in ("list", "structured"):
            merged[f] = []
        elif kind == "dict":
            merged[f] = {}

    for part in parts:
        for k, v in part.items():
            if k not in merged:
                continue
            if isinstance(merged[k], list) and isinstance(v, list):
                merged[k].extend(v)
            elif isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k].update(v)
            elif merged[k] in (None, ""):
                merged[k] = v
    return merged


def _extract_json_object(text: str) -> dict:
    """Robustly extract JSON object from LLM response."""
    def _repair_json_text(raw: str) -> str:
        # fix common LLM JSON mistakes without being too aggressive
        repaired = raw
        repaired = re.sub(r"\bNULL\b", "null", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r"\bNaN\b", "null", repaired)
        # insert missing commas between values and next key
        repaired = re.sub(r"([0-9\"\}\]])\s*(\")", r"\1, \2", repaired)
        return repaired

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json_text(cleaned))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json_text(match.group()))
        except json.JSONDecodeError:
            pass

    # try fixing trailing commas
    candidate = match.group()
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    candidate = _repair_json_text(candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def _clean_payload(data: dict, selected_fields: list[str]) -> dict:
    """Normalize extracted data to match schema types."""
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
            out[field] = text if text and text.lower() not in ("null", "none", "n/a") else None

    # skills_categorized  ensure it's a dict and clean keys/values
    if "skills_categorized" in selected_fields:
        raw_skills = data.get("skills_categorized", {})
        if isinstance(raw_skills, dict):
            out["skills_categorized"] = {
                str(k).strip(): [str(v).strip() for v in vals if v]
                for k, vals in raw_skills.items() if isinstance(vals, list)
            }
        else:
            out["skills_categorized"] = {}

    # data normalization (contact info)
    if out.get("phone"):
        # remove common artifacts and whitespace
        out["phone"] = re.sub(r'[^0-9+\-() ]', '', out["phone"]).strip()
    
    for f in ["linkedin", "github", "portfolio", "twitter"]:
        if out.get(f) and out[f].startswith("www."):
            out[f] = "https://" + out[f]

    # structured fields  pass through as-is (validators handle normalization)
    for field in STRUCTURED_FIELDS:
        if field in selected_fields and field in data:
            out[field] = data[field]

    return out


def _cache_key(
    resume_text: str,
    provider: str,
    model: str,
    selected_fields: list[str],
    truncation_limit: int,
) -> str:
    """Generate a deterministic cache key."""
    # hash based on the truncated text that will actually be seen by the LLM
    text_to_hash = resume_text[:truncation_limit]
    payload = (
        text_to_hash
        + "|" + provider
        + "|" + model
        + "|" + ",".join(sorted(selected_fields))
        + "|" + str(truncation_limit)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_from_cache(key: str) -> Optional[dict]:
    """Load cached result if it exists."""
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            with cache_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def _save_to_cache(key: str, data: dict) -> None:
    """Save result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    try:
        with cache_file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass


def extract_with_ollama(
    resume_text, model_name, host, selected_fields,
    extracted_links=None, truncation_limit=8000
):
    """Extract entities using local Ollama model."""
    global _last_error
    try:
        import requests
        prompt = _build_prompt(resume_text, selected_fields, fewshot=False,
                    extracted_links=extracted_links, truncation_limit=truncation_limit)
        template = _build_json_template(selected_fields)
        endpoint = host.rstrip("/") + "/api/chat"
        logger.info(f"Calling Ollama at {endpoint} with model {model_name}")
        # adapt generation options for faster response
        if truncation_limit <= 6000:
            num_predict = 256
            num_ctx = 1024
        else:
            num_predict = 512
            num_ctx = 2048

        # allow long runs for large inputs/models
        timeout_secs = 600

        payload = {
            "model": model_name,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

        # attempt request with a simple retry strategy on timeouts or transient errors
        duration = None
        try:
            t0 = time.time()
            response = requests.post(endpoint, json=payload, timeout=timeout_secs)
            duration = time.time() - t0
            logger.info(f"Ollama request finished in {duration:.2f}s (model={model_name}, predict={num_predict})")
            response.raise_for_status()
        except requests.exceptions.ReadTimeout:
            # retry with minimal generation settings  this helps when model is still loading
            try:
                logger.warning("Ollama read timeout; retrying with minimal generation settings")
                small_payload = payload.copy()
                small_payload["options"] = {"temperature": 0, "num_predict": 16, "num_ctx": 512}
                t0 = time.time()
                response = requests.post(endpoint, json=small_payload, timeout=max(60, timeout_secs * 2))
                duration = time.time() - t0
                logger.info(f"Ollama retry finished in {duration:.2f}s (model={model_name}, predict=16)")
                response.raise_for_status()
            except Exception as e2:
                _last_error = f"Ollama error after retry: {e2}"
                logger.error(_last_error)
                return None
        except requests.exceptions.RequestException as re:
            _last_error = f"Ollama request failed: {re}"
            logger.error(_last_error)
            return None

        def _ollama_content_from_response(resp) -> str:
            raw_text = (resp.text or "").strip()
            try:
                payload_json = resp.json()
            except Exception:
                payload_json = None
            if isinstance(payload_json, dict):
                message = payload_json.get("message", {})
                return str(message.get("content", "")).strip()

            # fallback: try to extract JSON object from raw response text
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if match:
                try:
                    obj = json.loads(match.group())
                    if isinstance(obj, dict) and "message" in obj:
                        message = obj.get("message", {})
                        return str(message.get("content", "")).strip()
                except Exception:
                    pass
            return raw_text

        text = _ollama_content_from_response(response)
        logger.info(f"Ollama response length: {len(text)} chars")

        logger.info("Ollama raw response length: %s", len(text))
        parsed = _extract_json_object(text)
        if not parsed:
            # retry once with a stricter prompt to force JSON only
            retry_prompt = (
                prompt
                                + "\n\nReturn ONLY a valid JSON object. No extra text, no markdown, no explanation."
                                    " Use ONLY the schema keys. Do not fabricate values."
                                + f"\n\nUse this exact template and fill values only:\n{template}"
            )
            payload["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": retry_prompt},
            ]
            try:
                response = requests.post(endpoint, json=payload, timeout=timeout_secs)
                response.raise_for_status()
                text_retry = _ollama_content_from_response(response)
                logger.info("Ollama retry response length: %s", len(text_retry))
                parsed = _extract_json_object(text_retry)
            except Exception as re_err:
                _last_error = f"Ollama retry failed: {re_err}"
                logger.error(_last_error)
                return None

        def _is_sparse(payload: dict) -> bool:
            if not payload:
                return True
            non_empty = 0
            for field in selected_fields:
                value = payload.get(field)
                if isinstance(value, list) and value:
                    non_empty += 1
                elif isinstance(value, dict) and value:
                    non_empty += 1
                elif value not in (None, "", []):
                    non_empty += 1
            return non_empty <= 1

        def _has_invalid_keys(payload: dict) -> bool:
            if not payload:
                return True
            allowed = set(selected_fields)
            keys = set(payload.keys())
            return len(keys - allowed) > 0

        if parsed and (_is_sparse(parsed) or _has_invalid_keys(parsed)):
            retry_prompt = (
                prompt
                + "\n\nYour last output was too empty. Extract as many fields as possible. "
                  "If a field exists in the text, do not return null or []."
                + f"\n\nUse this exact template and fill values only:\n{template}"
            )
            if _has_invalid_keys(parsed):
                retry_prompt += "\n\nRemove ALL keys that are not in the schema."
            payload["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": retry_prompt},
            ]
            try:
                response = requests.post(endpoint, json=payload, timeout=timeout_secs)
                response.raise_for_status()
                text_retry = _ollama_content_from_response(response)
                logger.info("Ollama sparse retry response length: %s", len(text_retry))
                parsed_retry = _extract_json_object(text_retry)
                if parsed_retry:
                    parsed = parsed_retry
            except Exception as re_err:
                logger.warning(f"Ollama sparse retry failed: {re_err}")

        def _enforce_schema(payload: dict) -> dict:
            # keep only schema keys and fill missing keys with null/[]/{} defaults
            def _default_for_field(field: str):
                kind = FIELD_SPECS.get(field, {}).get("kind", "string")
                if kind in ("list", "structured"):
                    return []
                if kind == "dict":
                    return {}
                return None

            fixed = {}
            for field in FIELD_SPECS.keys():
                if field in payload:
                    fixed[field] = payload[field]
                else:
                    fixed[field] = _default_for_field(field)
            return fixed

        if parsed:
            parsed = _enforce_schema(parsed)

        if not parsed:
            _last_error = f"Failed to parse JSON from Ollama response: {text[:200]}"
            logger.warning(_last_error)
            return None

        cleaned = _clean_payload(parsed, selected_fields)
        try:
            entity = ResumeEntity(**cleaned)
        except Exception as ve:
            logger.warning(f"Pydantic validation error, retrying without structured fields: {ve}")
            # drop structured fields and retry
            for sf in STRUCTURED_FIELDS:
                cleaned.pop(sf, None)
            entity = ResumeEntity(**cleaned)
        entity.backfill_flat_fields()
        _last_error = ""
        return entity
    except Exception as e:
        _last_error = f"Ollama error: {e}"
        logger.error(_last_error)
        return None


def extract_with_gemini(
    resume_text, api_key, model_name, selected_fields,
    extracted_links=None, truncation_limit=8000
):
    """Extract entities using Google Gemini API."""
    global _last_error
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        prompt = _build_prompt(resume_text, selected_fields, fewshot=False,
                    extracted_links=extracted_links, truncation_limit=truncation_limit)
        
        # simple retry logic for rate limits
        response = None
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                break
            except Exception as ge:
                if "429" in str(ge) and attempt < 2:
                    logger.warning(f"Gemini rate limit hit, retrying in {5 * (attempt + 1)}s...")
                    time.sleep(5 * (attempt + 1))
                else:
                    raise ge
        
        if not response:
            return None
            
        text = response.text.strip()

        parsed = _extract_json_object(text)
        if not parsed:
            _last_error = f"Failed to parse JSON from Gemini response: {text[:200]}"
            logger.warning(_last_error)
            return None

        cleaned = _clean_payload(parsed, selected_fields)
        try:
            entity = ResumeEntity(**cleaned)
        except Exception as ve:
            logger.warning(f"Pydantic validation error, retrying without structured fields: {ve}")
            for sf in STRUCTURED_FIELDS:
                cleaned.pop(sf, None)
            entity = ResumeEntity(**cleaned)
        entity.backfill_flat_fields()
        _last_error = ""
        return entity
    except Exception as e:
        _last_error = f"Gemini error: {e}"
        logger.error(_last_error)
        return None


def extract_entities(
    resume_text,
    provider="ollama-local",
    ollama_model=None,
    ollama_host=None,
    gemini_api_key=None,
    gemini_model=None,
    selected_fields=None,
    extracted_links=None,
    use_cache=True,
    truncation_limit=8000,
):
    """Unified extraction entry point for both providers.
    
    If use_cache is True and the same resume text + config has been seen before,
    returns the cached result instantly.
    """
    selected = _normalize_selected_fields(selected_fields)

    if provider != "ollama-local":
        logger.info("Gemini is disabled. Use Ollama.")
        return None

    model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not model:
        return None

    # check cache
    if use_cache:
        key = _cache_key(resume_text, provider, model, selected, truncation_limit)
        cached = _load_from_cache(key)
        if cached is not None:
            try:
                entity = ResumeEntity(**cached)
                logger.info("Cache hit for %s/%s", provider, model)
                return entity
            except Exception:
                pass  # cache corrupted, re-extract

    # section-wise extraction for higher accuracy (multiple calls)
    multi_section = True
    if multi_section:
        sections = _split_sections(resume_text)
        section_fields = {
            "contact": [
                "name", "email", "phone", "location", "linkedin",
                "github", "portfolio", "twitter", "other_links", "summary",
            ],
            "education": ["education_details", "college_name", "degree", "graduation_year", "certifications"],
            "experience": ["experience_details", "designation", "companies_worked_at", "years_of_experience"],
            "projects": ["projects_detailed", "projects"],
            "skills": ["skills", "skills_categorized", "languages"],
            "achievements": ["achievements", "publications", "hobbies", "references"],
            "activities": ["achievements", "hobbies"],
        }

        partials = []
        for sec, fields in section_fields.items():
            fields = [f for f in fields if f in selected]
            if not fields:
                continue
            text_chunk = sections.get(sec) or resume_text
            part_entity = extract_with_ollama(
                text_chunk, model, host, fields,
                extracted_links=extracted_links if sec == "contact" else None,
                truncation_limit=truncation_limit,
            )

            if part_entity is not None:
                partials.append(part_entity.model_dump())

        if partials:
            merged = _merge_section_results(selected, partials)
            if extracted_links and "other_links" in selected:
                merged.setdefault("other_links", [])
                for link in extracted_links:
                    if link not in merged["other_links"]:
                        merged["other_links"].append(link)
            if "projects_detailed" in merged:
                merged["projects_detailed"] = _map_project_links(merged["projects_detailed"], extracted_links)
            entity = ResumeEntity(**merged)
            if use_cache:
                key = _cache_key(resume_text, provider, model, selected, truncation_limit)
                _save_to_cache(key, entity.model_dump())
            return entity

    # call provider
    entity = extract_with_ollama(
        resume_text, model, host, selected,
        extracted_links=extracted_links,
        truncation_limit=truncation_limit
    )

    if entity is not None and extracted_links and "other_links" in selected:
        for link in extracted_links:
            if link not in entity.other_links:
                entity.other_links.append(link)
    if entity is not None and extracted_links:
        data = entity.model_dump()
        if "projects_detailed" in data:
            data["projects_detailed"] = _map_project_links(data["projects_detailed"], extracted_links)
            entity = ResumeEntity(**data)

    # save to cache
    if entity is not None and use_cache:
        key = _cache_key(resume_text, provider, model, selected, truncation_limit)
        _save_to_cache(key, entity.model_dump())

    return entity
