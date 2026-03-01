import os
import json
import csv
from collections import defaultdict
import statistics as stats

IN_DIR = "../../results/tcav_scores"
OUT_DIR = "../../results/tables"
OUT_CSV = os.path.join(OUT_DIR, "tcav_scores_table.csv")
OUT_TXT = os.path.join(OUT_DIR, "tcav_scores_summary.txt")

FILES = {
    "clean": os.path.join(IN_DIR, "clean_tcav_raw.json"),
    "spurious": os.path.join(IN_DIR, "spurious_tcav_raw.json"),
}

def load_rows(model_type, path):
    d = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    for exp_set, layer_dict in d["tcav_scores"].items():
        control_id = exp_set.split("-")[1] if "-" in exp_set else "unknown"
        for layer_name, obj in layer_dict.items():
            sign_patch = float(obj["sign_count"][0])
            sign_ctrl  = float(obj["sign_count"][1])
            mag_patch  = float(obj["magnitude"][0])
            mag_ctrl   = float(obj["magnitude"][1])

            rows.append({
                "model_type": model_type,
                "experimental_set": exp_set,
                "control_set_id": control_id,
                "layer": layer_name,
                "sign_count_patch": sign_patch,
                "sign_count_control": sign_ctrl,
                "magnitude_patch": mag_patch,
                "magnitude_control": mag_ctrl,
            })
    return rows

def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "model_type","experimental_set","control_set_id","layer",
        "sign_count_patch","sign_count_control","magnitude_patch","magnitude_control"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def summarize(rows):

    by = defaultdict(list)
    for r in rows:
        by[(r["model_type"], r["layer"])].append(r["sign_count_patch"])

    out_lines = []
    for (m, layer), vals in sorted(by.items()):
        out_lines.append(
            f"{m:8s} | {layer:6s} | patch_sign_count mean={stats.mean(vals):.4f} "
            f"std={stats.pstdev(vals):.4f} min={min(vals):.4f} max={max(vals):.4f} n={len(vals)}"
        )
    return out_lines

def main():
    all_rows = []
    for model_type, path in FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        all_rows.extend(load_rows(model_type, path))

    write_csv(all_rows, OUT_CSV)

    lines = []

    lines.extend(summarize(all_rows))

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[DONE] wrote:\n  {OUT_CSV}\n  {OUT_TXT}")

if __name__ == "__main__":
    main()