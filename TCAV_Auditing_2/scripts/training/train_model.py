import os
import time
import yaml
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


@dataclass
class TrainConfig:
    dataset_dir: str
    num_classes: int
    arch: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    num_workers: int
    save_path: str
    log_path: str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(cfg_path: str) -> TrainConfig:
    with open(cfg_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d["weight_decay"] = float(d["weight_decay"])
    d["lr"] = float(d["lr"])
    return TrainConfig(**d)


def build_model(arch: str, num_classes: int):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")


def log_line(path: str, msg: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_line(cfg.log_path, f"[INFO] device={device}")
    log_line(cfg.log_path, f"[INFO] cfg={cfg}")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root=cfg.dataset_dir, transform=train_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = build_model(cfg.arch, cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log_line(cfg.log_path, f"[E{epoch:03d}] loss={epoch_loss:.4f} acc={epoch_acc:.4f} lr={lr_now:.5f} time={dt:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg.__dict__},
                cfg.save_path
            )
            log_line(cfg.log_path, f"[SAVE] best_loss={best_loss:.4f} -> {cfg.save_path}")

    log_line(cfg.log_path, "[DONE]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to yaml config")
    args = ap.parse_args()
    main(args.cfg)