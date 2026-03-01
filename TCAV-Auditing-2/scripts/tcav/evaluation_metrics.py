import os
import csv
import statistics as stats

IN_CSV = "../../results/tcav_scores/tcav_scores_table.csv"
OUT_DIR = "../../results/tcav_scores"
OUT_DECISIONS = os.path.join(OUT_DIR, "tcav_detection_decisions.csv")
OUT_METRICS = os.path.join(OUT_DIR, "tcav_confusion_matrix.txt")

LAYER_SHALLOW = "layer1"
LAYER_DEEP = "layer3"
THRESHOLD = 0.30

GROUND_TRUTH = {
    "clean": 0,
    "spurious": 1,
}

def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["sign_count_patch"] = float(r["sign_count_patch"])
            r["sign_count_control"] = float(r["sign_count_control"])
            rows.append(r)
    return rows

def group_patch_scores(rows, model_type, layer_name):
    vals = []
    for r in rows:
        if r["model_type"] == model_type and r["layer"] == layer_name:
            vals.append(r["sign_count_patch"])
    if not vals:
        raise RuntimeError(f"No values found for model={model_type} layer={layer_name}")
    return vals

def decision_for_model(rows, model_type):
    shallow = group_patch_scores(rows, model_type, LAYER_SHALLOW)
    deep = group_patch_scores(rows, model_type, LAYER_DEEP)

    mean_shallow = stats.mean(shallow)
    mean_deep = stats.mean(deep)
    score = mean_deep - mean_shallow

    pred = 1 if score > THRESHOLD else 0
    return {
        "model_type": model_type,
        "layer_shallow": LAYER_SHALLOW,
        "layer_deep": LAYER_DEEP,
        "mean_patch_shallow": mean_shallow,
        "mean_patch_deep": mean_deep,
        "score_deep_minus_shallow": score,
        "threshold": THRESHOLD,
        "decision_rely_on_patch": pred,
        "ground_truth_rely_on_patch": GROUND_TRUTH.get(model_type, None),
    }

def confusion_matrix(decisions):
    tp = tn = fp = fn = 0
    for d in decisions:
        gt = d["ground_truth_rely_on_patch"]
        pred = d["decision_rely_on_patch"]
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 0 and pred == 0:
            tn += 1
    return tp, tn, fp, fn

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_rows(IN_CSV)

    model_types = sorted(set(r["model_type"] for r in rows))
    decisions = [decision_for_model(rows, m) for m in model_types]

    with open(OUT_DECISIONS, "w", newline="", encoding="utf-8") as f:
        cols = [
            "model_type",
            "mean_patch_shallow",
            "mean_patch_deep",
            "score_deep_minus_shallow",
            "threshold",
            "decision_rely_on_patch",
            "ground_truth_rely_on_patch",
            "layer_shallow",
            "layer_deep",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in decisions:
            w.writerow(d)

    tp, tn, fp, fn = confusion_matrix(decisions)

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("TCAV Detection Rule\n")
        f.write("===================\n")
        f.write(f"Rule: rely_on_patch = 1 if mean(patch_sign_count @ {LAYER_DEEP}) - mean(patch_sign_count @ {LAYER_SHALLOW}) > {THRESHOLD}\n\n")
        f.write("Per-model decisions:\n")
        for d in decisions:
            f.write(
                f"- {d['model_type']}: mean_deep={d['mean_patch_deep']:.4f}, mean_shallow={d['mean_patch_shallow']:.4f}, "
                f"score={d['score_deep_minus_shallow']:.4f} => pred={d['decision_rely_on_patch']} gt={d['ground_truth_rely_on_patch']}\n"
            )
        f.write("\nConfusion Matrix:\n")
        f.write(f"TP={tp}  FN={fn}\n")
        f.write(f"FP={fp}  TN={tn}\n")

    print(f"[DONE] wrote:\n  {OUT_DECISIONS}\n  {OUT_METRICS}")

if __name__ == "__main__":
    main()