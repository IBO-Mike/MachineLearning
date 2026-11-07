"""Re-evaluate the manuscript rule on archived scores, without training models."""

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "experiments/controlled/results/tables/tcav_scores_table.csv"
REFERENCE_LABELS = {"clean": 0, "spurious": 1}


def evaluate(rows):
    """Apply layer3 >= 0.7 on at least two of exactly three controls."""
    grouped = {model: {} for model in REFERENCE_LABELS}
    for row in rows:
        model = row["model_type"]
        if model not in grouped:
            raise ValueError(f"Unknown model: {model}")
        if row["layer"] != "layer3":
            continue
        control = row["control_set_id"]
        if control in grouped[model]:
            raise ValueError(f"Duplicate layer3 control for {model}: {control}")
        score = float(row["sign_count_patch"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score: {score}")
        grouped[model][control] = score

    decisions = []
    matrix = dict(TP=0, FP=0, TN=0, FN=0)
    for model, controls in grouped.items():
        if len(controls) != 3:
            raise ValueError(f"Expected three layer3 controls for {model}")
        passed = sum(score >= 0.7 for score in controls.values())
        prediction = int(passed >= 2)
        reference = REFERENCE_LABELS[model]
        outcome = ("TP" if prediction else "FN") if reference else ("FP" if prediction else "TN")
        matrix[outcome] += 1
        decisions.append(dict(model=model, layer3_scores=controls, controls_passed=passed,
                              prediction=prediction, reference_label=reference, outcome=outcome))
    return dict(rule="layer3 >= 0.7 for at least 2 of 3 controls",
                interpretation="Reference labels follow training conditions; not a causal oracle.",
                decisions=decisions, confusion_matrix=matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, help="Optional JSON output; default is stdout only")
    args = parser.parse_args()
    with args.table.open(encoding="utf-8", newline="") as stream:
        result = evaluate(list(csv.DictReader(stream)))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
