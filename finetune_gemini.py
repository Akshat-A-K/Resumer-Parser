# prepare and validate gemini finetuning via fewshot prompting

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FEWSHOT_PATH = Path("splits/fewshot_examples.json")
TRAIN_PATH = Path("splits/train.jsonl")


def load_fewshot_examples() -> list[dict]:
    if not FEWSHOT_PATH.exists():
        print(f"Error: {FEWSHOT_PATH} not found. Run finetune_prepare.py first.")
        sys.exit(1)
    with FEWSHOT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_train_data() -> list[dict]:
    if not TRAIN_PATH.exists():
        print(f"Error: {TRAIN_PATH} not found. Run finetune_prepare.py first.")
        sys.exit(1)
    rows = []
    with TRAIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_gemini_extraction(api_key: str, model_name: str, num_samples: int = 3) -> None:
    # test gemini extraction on a few samples to validate setup
    from llm_parser import extract_entities

    examples = load_fewshot_examples()
    test_samples = examples[:num_samples]

    print(f"\nTesting Gemini extraction on {len(test_samples)} samples...")
    print(f"Model: {model_name}")

    for i, sample in enumerate(test_samples, 1):
        text = sample.get("text", "")[:500]
        expected = sample.get("expected_output", {})

        print(f"\nSample {i}")
        print(f"Text preview: {text[:200]}...")
        print(f"Expected fields: {list(expected.keys())}")

        result = extract_entities(
            text,
            provider="gemini-api",
            gemini_api_key=api_key,
            gemini_model=model_name,
            finetuned=True,
        )

        if result:
            data = result.model_dump()
            extracted_fields = {k: v for k, v in data.items() if v is not None and v != []}
            print(f"Extracted fields: {list(extracted_fields.keys())}")
            print(f"Name: {data.get('name', 'N/A')}")
            print(f"Skills: {data.get('skills', [])[:5]}")
        else:
            print("Extraction FAILED")

    print("\nGemini validation complete.")


def prepare_tuning_data(train_data: list[dict], output_path: Path) -> int:
    # prepare training data in gemini tuning api format
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for sample in train_data:
            text = sample.get("text", "")
            expected = sample.get("expected_output", {})
            if not text or not expected:
                continue

            tuning_example = {
                "text_input": f"Extract resume entities from this text:\n---\n{text[:6000]}\n---",
                "output": json.dumps(expected, ensure_ascii=False),
            }
            f.write(json.dumps(tuning_example, ensure_ascii=False) + "\n")
            count += 1

    return count


def run_gemini_tuning(api_key: str, base_model: str, training_file: Path) -> None:
    # use gemini tuning api to create a finetuned model
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        training_data = []
        with training_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    training_data.append(json.loads(line))

        print(f"Submitting {len(training_data)} training examples to Gemini Tuning API...")
        print(f"Base model: {base_model}")

        operation = genai.create_tuned_model(
            display_name="resume-parser-tuned",
            source_model=base_model,
            training_data=training_data,
            epoch_count=5,
            batch_size=4,
            learning_rate=0.001,
        )

        print("Tuning job submitted. This may take several minutes...")
        for status in operation.wait_bar():
            pass

        result = operation.result()
        print(f"\nTuned model created: {result.name}")

    except Exception as e:
        print(f"Gemini Tuning API error: {e}")
        print("The fewshot prompting approach will still work without the Tuning API.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-samples", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-tuning-api", action="store_true")
    parser.add_argument("--tuning-base-model", default="models/gemini-1.5-flash-001-tuning")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY not set in .env file")
        sys.exit(1)

    model_name = args.model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    print("Preparing Gemini tuning data...")
    train_data = load_train_data()
    tuning_file = Path("splits/gemini_tuning_data.jsonl")
    count = prepare_tuning_data(train_data, tuning_file)
    print(f"Prepared {count} tuning examples at {tuning_file}")

    examples = load_fewshot_examples()
    print(f"Fewshot examples available: {len(examples)}")

    test_gemini_extraction(api_key, model_name, args.test_samples)

    if args.use_tuning_api:
        print("\nStarting Gemini Tuning API finetuning...")
        run_gemini_tuning(api_key, args.tuning_base_model, tuning_file)


if __name__ == "__main__":
    main()
