#!/usr/bin/env python3
"""Reconstruct v0.1 training data from the original competition CSV."""

import csv
import json
import pathlib


def main():
    csv_path = pathlib.Path(
        ".cache/kagglehub/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"
    )

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    output_path = pathlib.Path("data/v0.1_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    PROMPT_SUFFIX = (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )

    SYSTEM_PROMPT = (
        "You are a careful reasoning model. "
        "Solve the problem step by step and end with Final answer: \\boxed{...}."
    )

    count = 0
    with open(csv_path, newline="", encoding="utf-8") as in_fh, open(
        output_path, "w", encoding="utf-8"
    ) as out_fh:
        for row in csv.DictReader(in_fh):
            example_id = row["id"]
            problem = row["prompt"].strip()
            answer = row["answer"].strip()

            record = {
                "id": example_id,
                "prompt": problem + PROMPT_SUFFIX,
                "response": f"Final answer: \\boxed{{{answer}}}",
                "system": SYSTEM_PROMPT,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Reconstructed {count} v0.1 examples → {output_path}")


if __name__ == "__main__":
    main()
