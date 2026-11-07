"""Check the manuscript decision boundary and archived evidence consistency."""

import copy
import csv
import json
import unittest

from tools.paper_results import DEFAULT_TABLE, ROOT, evaluate


class PaperResultsTests(unittest.TestCase):
    def setUp(self):
        with DEFAULT_TABLE.open(encoding="utf-8", newline="") as stream:
            self.rows = list(csv.DictReader(stream))

    def test_archived_outcome(self):
        self.assertEqual(evaluate(self.rows)["confusion_matrix"], dict(TP=1, FP=1, TN=0, FN=0))

    def test_boundary_and_majority(self):
        rows = copy.deepcopy(self.rows)
        for model, values in [("clean", [0.7, 0.7, 0.0]), ("spurious", [0.7, 0.699, 0.0])]:
            selected = [r for r in rows if r["model_type"] == model and r["layer"] == "layer3"]
            for row, value in zip(selected, values):
                row["sign_count_patch"] = value
        result = evaluate(rows)
        self.assertEqual([d["prediction"] for d in result["decisions"]], [1, 0])

    def test_reject_missing_or_duplicate_controls(self):
        row = next(r for r in self.rows if r["layer"] == "layer3")
        with self.assertRaises(ValueError):
            evaluate(self.rows + [row])
        with self.assertRaises(ValueError):
            evaluate([r for r in self.rows if r is not row])

    def test_csv_matches_raw_json(self):
        self.assertEqual(len(self.rows), 24)
        for model in ("clean", "spurious"):
            path = ROOT / f"experiments/controlled/results/tcav_scores/{model}_tcav_raw.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            for row in self.rows:
                if row["model_type"] == model:
                    score = raw["tcav_scores"][row["experimental_set"]][row["layer"]]
                    for column, key, index in [("sign_count_patch", "sign_count", 0),
                                               ("sign_count_control", "sign_count", 1),
                                               ("magnitude_patch", "magnitude", 0),
                                               ("magnitude_control", "magnitude", 1)]:
                        self.assertEqual(float(row[column]), score[key][index])


if __name__ == "__main__":
    unittest.main()
