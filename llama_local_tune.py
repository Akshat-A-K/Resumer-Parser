"""Create a specialized local Ollama model for resume parsing.

Note: This is prompt-level specialization (custom model profile), not weight fine-tuning.

Usage:
    python llama_local_tune.py --base-model llama3.2:3b --new-model resume-parser-local
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

SYSTEM_PROMPT = """You are a strict resume parser.
Return only valid JSON.
Required keys:
name, email, location, designation, companies_worked_at, skills, college_name, degree, graduation_year, years_of_experience
Rules:
- string fields: string or null
- list fields: list of strings
- no explanation, no markdown
""".strip()


def resolve_ollama_executable() -> str:
    """Resolve Ollama CLI path, with Windows-friendly fallbacks."""
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
        "Ollama CLI not found. Install Ollama and ensure 'ollama' is on PATH, "
        "or install to one of the default locations."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="llama3.2:3b")
    parser.add_argument("--new-model", default="resume-parser-local")
    args = parser.parse_args()

    out_dir = Path("api_tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    modelfile = out_dir / "Modelfile.resume"

    modelfile.write_text(
        f"FROM {args.base_model}\n\nSYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"\n",
        encoding="utf-8",
    )

    ollama_exe = resolve_ollama_executable()

    cmd = [
        ollama_exe,
        "create",
        args.new_model,
        "-f",
        str(modelfile),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ollama create failed")

    print(f"Created local model: {args.new_model}")


if __name__ == "__main__":
    main()
