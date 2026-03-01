import os
import csv
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

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.numel()

        for cls in range(num_classes):
            mask = (y == cls)
            if mask.any():
                c_total[cls] += mask.sum().item()
                c_correct[cls] += (preds[mask] == y[mask]).sum().item()

    acc = correct / max(total, 1)
    c_acc = [c_correct[i]/max(c_total[i],1) for i in range(num_classes)]
    return acc, c_acc

def main():
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
    for model_name, model_path in experiments:
        for test_name, test_dir in testsets:
            acc, c_acc = eval_folder(model_path, test_dir)
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

    txt_path = "../../results/behavioral_tests/behavior_test_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['model']:14s} | {r['testset']:18s} | overall={r['overall_acc']:.4f} | target={r['target_class_acc']:.4f}\n")

    print(f"[DONE] wrote:\n  {csv_path}\n  {txt_path}")

if __name__ == "__main__":
    main()