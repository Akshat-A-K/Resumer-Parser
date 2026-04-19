# evaluate llm resume parsers on test split across 4 configurations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from evaluator import evaluate_batch, results_to_dataframe
from llm_parser import extract_entities
from schema import ResumeEntity

load_dotenv()

TEST_PATH = Path("splits/test.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


CONFIGS = {
    "ollama-direct": {"label": "Ollama Direct", "provider": "ollama-local", "finetuned": False},
    "ollama-finetuned": {"label": "Ollama Finetuned", "provider": "ollama-local", "finetuned": True},
    "gemini-direct": {"label": "Gemini Direct", "provider": "gemini-api", "finetuned": False},
    "gemini-finetuned": {"label": "Gemini Finetuned", "provider": "gemini-api", "finetuned": True},
}


def evaluate_config(config_name, config, test_rows, ollama_model, ollama_host, ollama_ft_model, gemini_model, sleep):
    # run evaluation for one configuration
    label = config["label"]
    provider = config["provider"]
    finetuned = config["finetuned"]

    if provider == "gemini-api":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print(f"  [{label}] Skipped, no GEMINI_API_KEY set")
            return None

    print(f"\n  Evaluating: {label}")
    print(f"  Provider: {provider} | Finetuned: {finetuned}")

    predictions = []
    ground_truths = []

    for idx, row in enumerate(test_rows, 1):
        text = row.get("text", "")
        expected = row.get("expected_output", {})

        if not text or not expected:
            continue

        gold = ResumeEntity(**expected)

        if provider == "ollama-local":
            model = ollama_ft_model if finetuned else ollama_model
        else:
            model = gemini_model

        pred = extract_entities(
            text,
            provider=provider,
            ollama_model=model if provider == "ollama-local" else None,
            ollama_host=ollama_host if provider == "ollama-local" else None,
            gemini_model=model if provider == "gemini-api" else None,
            finetuned=finetuned,
        )

        if pred is None:
            pred = ResumeEntity()

        predictions.append(pred)
        ground_truths.append(gold)

        if idx % 5 == 0 or idx == len(test_rows):
            print(f"  [{label}] Progress: {idx}/{len(test_rows)}")

        if sleep > 0:
            time.sleep(sleep)

    if not predictions:
        print(f"  [{label}] No predictions generated")
        return None

    metrics = evaluate_batch(predictions, ground_truths)

    macro = metrics.get("macro_avg", {})
    print(f"\n  [{label}] Results:")
    print(f"    Macro Precision: {macro.get('precision', 0):.4f}")
    print(f"    Macro Recall:    {macro.get('recall', 0):.4f}")
    print(f"    Macro F1:        {macro.get('f1', 0):.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--configs", default="all")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--ollama-ft-model", default="resume-parser-local")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()

    if not TEST_PATH.exists():
        print(f"Error: {TEST_PATH} not found. Run finetune_prepare.py first.")
        return

    test_rows = read_jsonl(TEST_PATH)
    if args.samples > 0:
        test_rows = test_rows[:args.samples]
    print(f"Test samples: {len(test_rows)}")

    if args.configs == "all":
        config_names = list(CONFIGS.keys())
    else:
        config_names = [c.strip() for c in args.configs.split(",")]

    ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_host = args.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    gemini_model = args.gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}

    for config_name in config_names:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}")
            continue

        config = CONFIGS[config_name]
        metrics = evaluate_config(
            config_name, config, test_rows,
            ollama_model, ollama_host, args.ollama_ft_model, gemini_model, args.sleep,
        )

        if metrics:
            all_results[config_name] = metrics
            json_path = output_dir / f"eval_{config_name}_{stamp}.json"
            json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            df = results_to_dataframe(metrics)
            csv_path = output_dir / f"eval_{config_name}_{stamp}.csv"
            df.to_csv(csv_path, index=False)

    if all_results:
        print(f"\nCOMPARISON SUMMARY")
        print(f"{'Config':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")

        best_config = None
        best_f1 = -1.0

        for name, metrics in all_results.items():
            macro = metrics.get("macro_avg", {})
            p = macro.get("precision", 0)
            r = macro.get("recall", 0)
            f1 = macro.get("f1", 0)
            label = CONFIGS[name]["label"]
            print(f"{label:<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_config = name

        if best_config:
            print(f"\nBest: {CONFIGS[best_config]['label']} (F1={best_f1:.4f})")

        report_path = output_dir / f"comparison_{stamp}.json"
        report_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
