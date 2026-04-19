# create specialized ollama model with system prompt and fewshot examples

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from schema import FIELD_SPECS

FEWSHOT_PATH = Path("splits/fewshot_examples.json")
OUT_DIR = Path("api_tuning")

SYSTEM_PROMPT = '''You are an expert resume parser that extracts structured information from resume text.

STRICT RULES:
1. Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
2. Use EXACTLY these keys in your output:
{field_list}
3. For string fields: return the extracted value as a string, or null if not found.
4. For list fields: return a JSON array of strings, or [] if not found.
5. Do NOT invent or hallucinate information. Only extract what is present.
6. Extract ALL skills mentioned (technical, tools, frameworks, methodologies, soft skills).
7. Extract ALL job titles and company names mentioned (current and past roles).
8. For URLs: extract full links for LinkedIn, GitHub, Twitter, portfolio, and any other URLs.
9. For education: extract all institution names, degrees, and graduation years.
10. For experience: compute total years if explicitly stated.
11. Be thorough and comprehensive. Extract EVERY relevant piece of information.'''


def build_field_list() -> str:
    lines = []
    for field, spec in FIELD_SPECS.items():
        kind = "list of strings" if spec["kind"] == "list" else "string or null"
        lines.append(f'   - "{field}": {kind}  ({spec["label"]})')
    return "\n".join(lines)


def load_fewshot_examples() -> list[dict]:
    if not FEWSHOT_PATH.exists():
        print(f"Warning: {FEWSHOT_PATH} not found. Run finetune_prepare.py first.")
        return []
    with FEWSHOT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_modelfile(base_model: str, examples: list[dict]) -> str:
    # build ollama modelfile with system prompt and fewshot messages
    field_list = build_field_list()
    system = SYSTEM_PROMPT.replace("{field_list}", field_list)

    lines = [
        f"FROM {base_model}",
        "",
        "PARAMETER temperature 0",
        "PARAMETER top_p 0.9",
        "PARAMETER num_predict 4096",
        "PARAMETER stop </s>",
        "",
        f'SYSTEM """{system}"""',
    ]

    # embed fewshot examples as conversation turns
    for i, ex in enumerate(examples[:5]):
        text_snippet = str(ex.get("text", ""))[:1500]
        expected = ex.get("expected_output", {})
        expected_json = json.dumps(expected, indent=2, ensure_ascii=False)

        lines.append("")
        lines.append(f'MESSAGE user """Extract resume entities from this text:\n---\n{text_snippet}\n---"""')
        lines.append(f'MESSAGE assistant """{expected_json}"""')

    return "\n".join(lines)


def resolve_ollama_executable() -> str:
    found = shutil.which("ollama")
    if found:
        return found

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Ollama CLI not found. Install Ollama and ensure ollama is on PATH."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="llama3.2:3b")
    parser.add_argument("--new-model", default="resume-parser-local")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = load_fewshot_examples()
    print(f"Loaded {len(examples)} fewshot examples")

    modelfile_content = build_modelfile(args.base_model, examples)
    modelfile_path = OUT_DIR / "Modelfile.resume"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"Modelfile written to {modelfile_path}")

    ollama_exe = resolve_ollama_executable()
    cmd = [ollama_exe, "create", args.new_model, "-f", str(modelfile_path)]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        error_msg = result.stderr.strip() or "ollama create failed"
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)

    print(f"Successfully created model: {args.new_model}")


if __name__ == "__main__":
    main()
