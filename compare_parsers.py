# compare resume parser configurations on labeled dataset

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from evaluator import annotations_to_entity, evaluate_batch, load_ground_truth, results_to_dataframe
from llm_parser import extract_entities
from schema import ResumeEntity

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=220)
    parser.add_argument("--dataset", default="Entity Recognition in Resumes.json")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--ollama-ft-model", default="resume-parser-local")
    parser.add_argument("--gemini-model", default=None)
    return parser.parse_args()


MODES = {
    "ollama-direct": {"label": "Ollama Direct", "provider": "ollama-local", "finetuned": False},
    "ollama-finetuned": {"label": "Ollama Finetuned", "provider": "ollama-local", "finetuned": True},
    "gemini-direct": {"label": "Gemini Direct", "provider": "gemini-api", "finetuned": False},
    "gemini-finetuned": {"label": "Gemini Finetuned", "provider": "gemini-api", "finetuned": True},
}


def evaluate_mode(mode_name, mode_config, rows, ollama_model, ollama_host, ollama_ft_model, gemini_model, sleep):
    label = mode_config["label"]
    provider = mode_config["provider"]
    finetuned = mode_config["finetuned"]

    if provider == "gemini-api" and not os.getenv("GEMINI_API_KEY", ""):
        print(f"[{label}] Skipped, no GEMINI_API_KEY")
        return None

    predictions = []
    ground_truths = []

    for index, sample in enumerate(rows, start=1):
        content = sample.get("content", "")
        annotations = sample.get("annotation", [])
        if not content or not annotations:
            continue

        gold = annotations_to_entity(annotations)

        if provider == "ollama-local":
            model = ollama_ft_model if finetuned else ollama_model
        else:
            model = gemini_model

        pred = extract_entities(
            content,
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

        if index % 10 == 0 or index == len(rows):
            print(f"[{label}] Progress: {index}/{len(rows)}")
        if sleep > 0:
            time.sleep(sleep)

    if not predictions:
        return None

    metrics = evaluate_batch(predictions, ground_truths)
    return metrics


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_ground_truth(str(dataset_path))
    rows = rows[: max(1, min(args.samples, len(rows)))]

    if args.mode == "all":
        mode_names = list(MODES.keys())
    else:
        mode_names = [m.strip() for m in args.mode.split(",")]

    ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_host = args.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    gemini_model = args.gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {}

    for mode_name in mode_names:
        if mode_name not in MODES:
            print(f"Unknown mode: {mode_name}")
            continue

        print(f"\nStarting mode: {mode_name}")
        metrics = evaluate_mode(
            mode_name, MODES[mode_name], rows,
            ollama_model, ollama_host, args.ollama_ft_model, gemini_model, args.sleep,
        )

        if metrics is None:
            continue

        df = results_to_dataframe(metrics)
        csv_path = output_dir / f"comparison_{mode_name}_{stamp}.csv"
        json_path = output_dir / f"comparison_{mode_name}_{stamp}.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        macro = metrics.get("macro_avg", {})
        print(
            f"[{mode_name}] Macro P={macro.get('precision', 0.0):.4f} "
            f"R={macro.get('recall', 0.0):.4f} F1={macro.get('f1', 0.0):.4f}"
        )

        report[mode_name] = {
            "label": MODES[mode_name]["label"],
            "macro_precision": macro.get("precision", 0.0),
            "macro_recall": macro.get("recall", 0.0),
            "macro_f1": macro.get("f1", 0.0),
        }

    if report:
        print(f"\nCOMPARISON SUMMARY")
        print(f"{'Mode':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")

        for name, data in report.items():
            print(f"{data['label']:<25} {data['macro_precision']:>10.4f} "
                  f"{data['macro_recall']:>10.4f} {data['macro_f1']:>10.4f}")

        best = max(report.items(), key=lambda kv: kv[1]["macro_f1"])
        print(f"\nBest: {best[1]['label']} (F1={best[1]['macro_f1']:.4f})")

        report_path = output_dir / f"comparison_summary_{stamp}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
