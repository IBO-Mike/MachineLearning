import os
import json
import random
import tempfile
from hashlib import sha256
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV, Concept
from torch.utils.data import DataLoader
from captum.concept._utils.common import concepts_to_str

@dataclass
class TcavConfig:
    clean_ckpt: str = "../../models/clean_model/model.pth"
    spurious_ckpt: str = "../../models/spurious_model/model.pth"

    patch_concept_dir: str = "../../data/patch_set/patch"
    control_sets_root: str = "../../data/control_sets"

    audit_inputs_dir: str = "../../data/test_clean/3"
    audit_target_class_idx: int = 3

    layers: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")
    max_audit_images: int = 200
    concept_batch_size: int = 32

    out_dir: str = "../../results/tcav_scores"
    seed: int = 42
    cav_seed: int = 42

def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_default_device("cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))

def build_resnet18(num_classes: int = 10) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_model(ckpt_path: str) -> nn.Module:
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_resnet18(num_classes=10)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def img_to_tensor(path: str, tfm) -> torch.Tensor:
    with Image.open(path) as image:
        return tfm(image.convert("RGB"))

def image_paths(folder):
    files = sorted(p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if not files:
        raise ValueError(f"No images in {folder}")
    return files


def manifest(paths):
    return [{"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()} for path in paths]


def make_concept(name, cid, paths, tfm, batch_size):
    tensors = [img_to_tensor(path, tfm) for path in paths]
    return Concept(id=cid, name=name, data_iter=DataLoader(tensors, batch_size=batch_size))


def run_tcav_for_model(
    model_name: str,
    model: nn.Module,
    cfg: TcavConfig,
) -> Dict:

    random.seed(cfg.cav_seed)
    np.random.seed(cfg.cav_seed)
    torch.manual_seed(cfg.cav_seed)
    tfm = transforms.Compose([transforms.ToTensor()])

    patch_paths = image_paths(cfg.patch_concept_dir)
    patch_concept = make_concept("patch", 0, patch_paths, tfm, cfg.concept_batch_size)
    control_folders = sorted(path for path in Path(cfg.control_sets_root).iterdir() if path.is_dir())
    if len(control_folders) != 3:
        raise ValueError(f"Expected three control folders, found {len(control_folders)}")

    experimental_sets = []
    controls = {}
    for cid, folder in enumerate(control_folders, start=1):
        paths = image_paths(folder)
        control = make_concept(folder.name, cid, paths, tfm, cfg.concept_batch_size)
        experimental_sets.append([patch_concept, control])
        controls[folder.name] = manifest(paths)

    audit_paths = image_paths(cfg.audit_inputs_dir)
    random.Random(cfg.seed).shuffle(audit_paths)
    audit_paths = audit_paths[:cfg.max_audit_images]
    audit_inputs = torch.stack([img_to_tensor(path, tfm) for path in audit_paths])

    lig = LayerIntegratedGradients(model, None, multiply_by_inputs=False)
    # Each model invocation starts with an empty cache, including after a retry.
    with tempfile.TemporaryDirectory(dir=cfg.out_dir, prefix="captum-") as cache:
        tcav = TCAV(model=model, layers=list(cfg.layers), layer_attr_method=lig,
                    model_id=model_name, save_path=cache + os.sep)
        tcav_scores = tcav.interpret(inputs=audit_inputs, experimental_sets=experimental_sets,
                                     target=cfg.audit_target_class_idx)

    out = {
        "model_name": model_name,
        "checkpoint_sha256": sha256(Path(
            cfg.clean_ckpt if model_name == "clean" else cfg.spurious_ckpt
        ).read_bytes()).hexdigest(),
        "sampling_seed": cfg.seed,
        "cav_seed": cfg.cav_seed,
        "transform": "RGB + ToTensor",
        "samples": {"patch": manifest(patch_paths), "controls": controls,
                    "audit": manifest(audit_paths)},
        "layers": list(cfg.layers),
        "audit_inputs_dir": cfg.audit_inputs_dir,
        "audit_target_class_idx": cfg.audit_target_class_idx,
        "patch_concept_dir": cfg.patch_concept_dir,
        "control_sets_root": cfg.control_sets_root,
        "tcav_scores": tcav_scores,
        "experimental_sets": [concepts_to_str(es) for es in experimental_sets],
    }
    return out

def flatten_tcav_scores(model_name: str, tcav_result: Dict) -> List[Dict]:
    scores = tcav_result["tcav_scores"]
    rows = []

    for exp_set_str, layer_dict in scores.items():
        parts = exp_set_str.split("-")
        control_name = parts[1] if len(parts) > 1 else "unknown_control"

        for layer_name, score_obj in layer_dict.items():
            rows.append({
                "model_type": model_name,
                "layer": layer_name,
                "control_set_id": control_name,
                "tcav_sign_count_score": float(score_obj["sign_count"][0]),
                "tcav_magnitude_score": float(score_obj["magnitude"][0]),
            })
    return rows

def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    else:
        return tensor_to_serializable(obj)

def save_outputs(cfg: TcavConfig, model_name: str, tcav_result: Dict, rows: List[Dict]):
    os.makedirs(cfg.out_dir, exist_ok=True)

    raw_path = os.path.join(cfg.out_dir, f"{model_name}_tcav_raw.json")
    safe_result = make_json_safe(tcav_result)

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(safe_result, f, indent=2)

    flat_path = os.path.join(cfg.out_dir, f"{model_name}_tcav_flat.jsonl")
    with open(flat_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[DONE] saved:\n  {raw_path}\n  {flat_path}")

def main():
    force_cpu()
    cfg = TcavConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    clean_model = load_model(cfg.clean_ckpt)
    clean_out = run_tcav_for_model("clean", clean_model, cfg)
    clean_rows = flatten_tcav_scores("clean", clean_out)
    save_outputs(cfg, "clean", clean_out, clean_rows)

    spurious_model = load_model(cfg.spurious_ckpt)
    spurious_out = run_tcav_for_model("spurious", spurious_model, cfg)
    spurious_rows = flatten_tcav_scores("spurious", spurious_out)
    save_outputs(cfg, "spurious", spurious_out, spurious_rows)


if __name__ == "__main__":
    main()
