"""Run the fixed, paired-seed CIFAR-10 shortcut protocol and save per-model evidence."""

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median, stdev
import sys

import captum
import numpy as np
from PIL import Image
from scipy.stats import binomtest, spearmanr
import sklearn
import torch
from torch import nn
from torchvision.datasets import CIFAR10

from experiments.controlled.scripts.tcav.run_tcav import (
    TcavConfig, flatten_tcav_scores, run_tcav_for_model, save_outputs,
)
from tools.paper_results import evaluate

SEEDS = (101, 102, 103, 104, 105)
EPOCHS = 10
BATCH = 256
BASELINE_THRESHOLD = 0.10


@dataclass(frozen=True)
class Scenario:
    name: str
    target: int


SCENARIOS = (Scenario("cat-red", 3), Scenario("airplane-checker", 0))


class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self.block(3, 24)
        self.layer2 = self.block(24, 48)
        self.layer3 = self.block(48, 64)
        self.layer4 = self.block(64, 64, stride=1)
        self.classifier = nn.Linear(64 * 2 * 2, 10)

    @staticmethod
    def block(inputs, outputs, stride=2):
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.classifier(nn.functional.avg_pool2d(x, 2).flatten(1))


def patch(images, scenario):
    edited = images.copy()
    if scenario.name == "cat-red":
        edited[:, -8:, -8:, :] = (255, 0, 0)
    else:
        tiles = np.indices((8, 8)) // 2
        checker = ((tiles[0] + tiles[1]) % 2 * 255).astype(np.uint8)
        edited[:, :8, :8, :] = checker[..., None]
    return edited


def tensors(images):
    return torch.from_numpy(images).permute(0, 3, 1, 2)


def fit(images, labels, seed, path):
    torch.manual_seed(seed)
    model = CompactCNN().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    history = []
    for epoch in range(EPOCHS):
        correct = 0
        loss_total = 0.0
        for indices in torch.randperm(len(labels)).split(BATCH):
            x = images[indices].float().div_(255).contiguous(memory_format=torch.channels_last)
            y = labels[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        scheduler.step()
        history.append({"loss": loss_total / len(labels), "accuracy": correct / len(labels)})
        print(f"{path.stem} epoch {epoch + 1}/{EPOCHS}: acc={history[-1]['accuracy']:.3f}", flush=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"state": model.state_dict(), "history": history}, temporary)
    temporary.replace(path)


@torch.inference_mode()
def predict(model, images):
    return torch.cat([
        model(batch.float().div_(255).contiguous(memory_format=torch.channels_last))
        for batch in tensors(images).split(BATCH)
    ])


def behavior(original, edited, labels, target):
    target_mask = labels == target
    before = original.argmax(1) == target
    after = edited.argmax(1) == target
    delta = (edited.softmax(1) - original.softmax(1))[:, target]
    logit = (edited - original)[:, target]
    change = after.int() - before.int()
    attraction = float(change[~target_mask].float().mean())
    return {
        "original_accuracy": float((original.argmax(1) == labels).float().mean()),
        "target_accuracy": float(before[target_mask].float().mean()),
        "patched_target_accuracy": float(after[target_mask].float().mean()),
        "target_probability_change": float(delta[target_mask].mean()),
        "target_logit_change": float(logit[target_mask].mean()),
        "non_target_probability_change": float(delta[~target_mask].mean()),
        "non_target_logit_change": float(logit[~target_mask].mean()),
        "non_target_attraction": attraction,
        "non_target_gains": int((change[~target_mask] > 0).sum()),
        "non_target_losses": int((change[~target_mask] < 0).sum()),
        "intervention_flag": int(attraction >= BASELINE_THRESHOLD),
    }


def audit_files(directory, train, test, scenario):
    # The concept and three controls draw from disjoint training indices.
    # Both conditions and all model seeds share the same input files.
    rng = np.random.default_rng(42)
    indices = rng.choice(len(train.data), 800, replace=False).reshape(4, 200)
    candidates = np.flatnonzero(np.asarray(test.targets) == scenario.target)
    audit = rng.choice(candidates, 200, replace=False)
    sets = {"concept": (indices[0], patch(train.data[indices[0]], scenario)),
            "audit": (audit, test.data[audit])}
    for i, selection in enumerate(indices[1:], start=1):
        sets[f"controls/{i}"] = (selection, train.data[selection])
    for name, (ids, images) in sets.items():
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        for identifier, image in zip(ids, images):
            Image.fromarray(image).save(target / f"{identifier:05d}.png")


def summarize(rows):
    summary = {}
    for scenario in SCENARIOS:
        group = [row for row in rows if row["scenario"] == scenario.name]
        scenario_summary = {}
        for key, reference in (("fpr", "clean"), ("tpr", "spurious")):
            models = [r for r in group if r["condition"] == reference]
            scenario_summary[key] = {}
            for detector in ("tcav_flag", "intervention_flag"):
                hits = sum(r[detector] for r in models)
                ci = binomtest(hits, len(models)).proportion_ci(method="exact")
                scenario_summary[key][detector] = [hits, len(models), ci.low, ci.high]
            metric_names = ("original_accuracy", "target_accuracy", "patched_target_accuracy",
                            "target_logit_change", "non_target_logit_change",
                            "non_target_probability_change", "non_target_attraction")
            scenario_summary[reference] = {
                metric: [mean(r[metric] for r in models), stdev(r[metric] for r in models)]
                for metric in metric_names
            }
        tcav = [r["median_tcav"] for r in group]
        effects = [r["non_target_probability_change"] for r in group]
        scenario_summary["descriptive_spearman"] = (
            float(spearmanr(tcav, effects).statistic)
            if len(set(tcav)) > 1 and len(set(effects)) > 1 else None
        )
        summary[scenario.name] = scenario_summary
    return summary


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": "CIFAR-10",
        "seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH,
        "architecture": "CompactCNN", "sample_and_cav_seed": 42,
        "tcav_rule": "layer3 >= 0.7 in at least 2 of 3 contrasts",
        "baseline": "net non-target prediction attraction >= 0.10",
        "scenarios": [vars(s) for s in SCENARIOS],
        "environment": {
            "python": sys.version, "torch": str(torch.__version__),
            "captum": captum.__version__, "sklearn": sklearn.__version__, "threads": 6,
        },
    }
    protocol_file = out / "protocol.json"
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("Use a new output directory when the protocol changes.")
    else:
        write_json(protocol_file, protocol)

    torch.set_num_threads(6)
    train = CIFAR10(data, train=True)
    test = CIFAR10(data, train=False)
    train_labels = torch.tensor(train.targets)
    test_labels = torch.tensor(test.targets)
    clean_train = tensors(train.data)
    scenario_train = {}
    for scenario in SCENARIOS:
        audit_files(out / "inputs" / scenario.name, train, test, scenario)
        images = train.data.copy()
        mask = train_labels.numpy() == scenario.target
        images[mask] = patch(images[mask], scenario)
        scenario_train[scenario.name] = tensors(images)

    rows = []
    for seed in SEEDS:
        clean_path = out / f"clean-{seed}.pth"
        if not clean_path.exists():
            fit(clean_train, train_labels, seed, clean_path)
        for scenario in SCENARIOS:
            shortcut_path = out / f"{scenario.name}-{seed}.pth"
            if not shortcut_path.exists():
                fit(scenario_train[scenario.name], train_labels, seed, shortcut_path)
            case_dir = out / scenario.name / str(seed)
            case_dir.mkdir(parents=True, exist_ok=True)
            inputs = out / "inputs" / scenario.name
            cfg = TcavConfig(
                clean_ckpt=str(clean_path), spurious_ckpt=str(shortcut_path),
                patch_concept_dir=str(inputs / "concept"),
                control_sets_root=str(inputs / "controls"),
                audit_inputs_dir=str(inputs / "audit"),
                audit_target_class_idx=scenario.target,
                layers=("layer3",), out_dir=str(case_dir),
            )
            pair = []
            score_rows = []
            for condition, checkpoint in (("clean", clean_path), ("spurious", shortcut_path)):
                model = CompactCNN().to(memory_format=torch.channels_last)
                model.load_state_dict(torch.load(checkpoint, weights_only=True)["state"])
                model.eval()
                evidence = behavior(predict(model, test.data), predict(model, patch(test.data, scenario)),
                                    test_labels, scenario.target)
                result = run_tcav_for_model(condition, model, cfg)
                scores = flatten_tcav_scores(condition, result)
                save_outputs(cfg, condition, result, scores)
                score_rows.extend({**score, "sign_count_patch": score["tcav_sign_count_score"]}
                                  for score in scores)
                ordered = sorted(scores, key=lambda score: score["control_set_id"])
                values = [score["tcav_sign_count_score"] for score in ordered]
                pair.append({"scenario": scenario.name, "seed": seed, "condition": condition,
                             **evidence, **{f"control_{i}": v for i, v in enumerate(values, 1)},
                             "median_tcav": median(values)})
            for row, decision in zip(pair, evaluate(score_rows)["decisions"], strict=True):
                assert row["condition"] == decision["model"]
                row["tcav_flag"] = decision["prediction"]
            rows.extend(pair)
            write_json(case_dir / "paired_metrics.json", pair)
            print(f"Evaluated {scenario.name}, seed {seed}", flush=True)

    with (out / "model_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "summary.json", summarize(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
