import os
import csv
from pathlib import Path
from statistics import mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

TARGET_CLASS = 3

def build_model(num_classes=10):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def eval_folder(model_path: str, data_dir: str, batch_size=256, num_workers=0):
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)

    model = build_model(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.ImageFolder(root=data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    correct = 0
    total = 0

    num_classes = 10
    c_correct = [0]*num_classes
    c_total = [0]*num_classes

    cat_outputs = {}
    offset = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.numel()
        for index, prediction, logit in zip(
            (y == TARGET_CLASS).nonzero(as_tuple=True)[0].tolist(),
            preds[y == TARGET_CLASS].tolist(),
            logits[y == TARGET_CLASS, TARGET_CLASS].tolist(),
        ):
            path, _ = ds.samples[offset + index]
            sample_id = Path(path).relative_to(data_dir).as_posix()
            cat_outputs[sample_id] = (prediction, logit)
        offset += y.numel()

        for cls in range(num_classes):
            mask = (y == cls)
            if mask.any():
                c_total[cls] += mask.sum().item()
                c_correct[cls] += (preds[mask] == y[mask]).sum().item()

    acc = correct / max(total, 1)
    c_acc = [c_correct[i]/max(c_total[i],1) for i in range(num_classes)]
    return acc, c_acc, cat_outputs

def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs("../../results/behavioral_tests", exist_ok=True)

    experiments = [
        ("clean_model",   "../../models/clean_model/model.pth"),
        ("spurious_model","../../models/spurious_model/model.pth"),
    ]

    testsets = [
        ("test_clean", "../../data/test_clean"),
        ("test_patched_target", "../../data/test_patched_target"),
        ("test_patched_then_removed", "../../data/test_patched_then_removed"),
    ]

    rows = []
    paired_rows = []
    for model_name, model_path in experiments:
        original = None
        for test_name, test_dir in testsets:
            acc, c_acc, outputs = eval_folder(model_path, test_dir)
            if original is None:
                if not outputs:
                    raise ValueError("No target-class images")
                original = outputs
            else:
                if outputs.keys() != original.keys():
                    raise ValueError(f"Target image IDs differ for {test_name}")
                gains = losses = 0
                logit_changes = []
                for image_id, (prediction, logit) in outputs.items():
                    base_prediction, base_logit = original[image_id]
                    gains += base_prediction != TARGET_CLASS and prediction == TARGET_CLASS
                    losses += base_prediction == TARGET_CLASS and prediction != TARGET_CLASS
                    logit_changes.append(logit - base_logit)
                paired_rows.append({
                    "model": model_name, "testset": test_name, "n_cat": len(outputs),
                    "gained_cat_predictions": gains, "lost_cat_predictions": losses,
                    "cat_accuracy_change": (gains - losses) / len(outputs),
                    "mean_cat_logit_change": mean(logit_changes),
                })
            rows.append({
                "model": model_name,
                "testset": test_name,
                "overall_acc": acc,
                "target_class_acc": c_acc[TARGET_CLASS],
            })

    csv_path = "../../results/behavioral_tests/behavior_test_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","testset","overall_acc","target_class_acc"])
        w.writeheader()
        w.writerows(rows)

    paired_path = "../../results/behavioral_tests/paired_target_summary.csv"
    with open(paired_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    txt_path = "../../results/behavioral_tests/behavior_test_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['model']:14s} | {r['testset']:18s} | overall={r['overall_acc']:.4f} | target={r['target_class_acc']:.4f}\n")

    print(f"[DONE] wrote:\n  {csv_path}\n  {txt_path}")

if __name__ == "__main__":
    main()