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
from tqdm import tqdm


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

    log_every_steps: int = 50
    tqdm_update_every_steps: int = 20


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

    d["lr"] = float(d["lr"])
    d["weight_decay"] = float(d["weight_decay"])

    d.setdefault("log_every_steps", 50)
    d.setdefault("tqdm_update_every_steps", 20)

    return TrainConfig(**d)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")


def log_line(path: str, msg: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


@torch.no_grad()
def quick_sanity_check_dataset(ds: datasets.ImageFolder, log_path: str):
    n = len(ds)
    log_line(log_path, f"[INFO] dataset_size={n}")
    if n == 0:
        raise RuntimeError("Dataset is empty. Check data directory and ImageFolder structure.")

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(cfg.seed)
    device = get_device()

    log_line(cfg.log_path, f"[INFO] device={device}")
    log_line(cfg.log_path, f"[INFO] cfg={cfg}")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root=cfg.dataset_dir, transform=train_tf)
    quick_sanity_check_dataset(train_ds, cfg.log_path)

    use_pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        persistent_workers=(cfg.num_workers > 0),
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

    total_steps_per_epoch = len(train_loader)
    if total_steps_per_epoch == 0:
        raise RuntimeError("DataLoader has 0 steps. Check dataset and batch_size.")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            total=total_steps_per_epoch,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for step, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device, non_blocking=use_pin), y.to(device, non_blocking=use_pin)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += bs

            avg_loss = running_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            lr_now = optimizer.param_groups[0]["lr"]

            if step == 1 or step % cfg.tqdm_update_every_steps == 0 or step == total_steps_per_epoch:
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    acc=f"{avg_acc:.4f}",
                    lr=f"{lr_now:.5f}",
                    step=f"{step}/{total_steps_per_epoch}",
                )

            if step == 1 or step % cfg.log_every_steps == 0 or step == total_steps_per_epoch:
                log_line(
                    cfg.log_path,
                    f"[E{epoch:03d} S{step:04d}] loss={avg_loss:.4f} acc={avg_acc:.4f} lr={lr_now:.5f}"
                )

        scheduler.step()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
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