import os
import glob
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV, Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

@dataclass
class TcavConfig:
    clean_ckpt: str = "models/clean_model/model.pth"
    spurious_ckpt: str = "models/spurious_model/model.pth"

    patch_concept_dir: str = "data/patch_set/patch"
    control_sets_root: str = "data/control_sets"

    audit_inputs_dir: str = "data/test_clean/3"
    audit_target_class_idx: int = 3

    layers: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")
    max_audit_images: int = 200
    concept_batch_size: int = 32
    audit_batch_size: int = 64

    out_dir: str = "results/tcav_scores"
    seed: int = 42

def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_default_device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

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
    img = Image.open(path).convert("RGB")
    return tfm(img)

def make_concept(name: str, cid: int, folder: str, tfm, batch_size: int) -> Concept:
    folder = os.path.join(folder, "")
    dataset = CustomIterableDataset(lambda fn: img_to_tensor(fn, tfm), folder)
    concept_iter = dataset_to_dataloader(dataset, batch_size=batch_size)
    return Concept(id=cid, name=name, data_iter=concept_iter)

def list_control_folders(control_root: str) -> List[str]:
    subdirs = [d for d in glob.glob(os.path.join(control_root, "*")) if os.path.isdir(d)]
    subdirs.sort()
    return subdirs

def load_audit_tensors(audit_dir: str, tfm, max_n: int, seed: int) -> torch.Tensor:
    files = []
    for ext in ("png", "jpg", "jpeg", "bmp", "webp"):
        files.extend(glob.glob(os.path.join(audit_dir, f"*.{ext}")))
    if len(files) == 0:
        raise RuntimeError(f"No images found under audit_inputs_dir: {audit_dir}")

    random.Random(seed).shuffle(files)
    files = files[:max_n]

    tensors = [img_to_tensor(p, tfm) for p in files]
    return torch.stack(tensors, dim=0)

def run_tcav_for_model(
    model_name: str,
    model: nn.Module,
    cfg: TcavConfig,
) -> Dict:

    tfm = transforms.Compose([transforms.ToTensor()])

    patch_concept = make_concept(
        name="patch",
        cid=0,
        folder=cfg.patch_concept_dir,
        tfm=tfm,
        batch_size=cfg.concept_batch_size,
    )

    control_folders = list_control_folders(cfg.control_sets_root)
    if len(control_folders) == 0:
        raise RuntimeError(f"No control set folders found under: {cfg.control_sets_root}")

    control_concepts = []
    for i, folder in enumerate(control_folders, start=1):
        cname = os.path.basename(folder.rstrip("/"))
        control_concepts.append(
            make_concept(
                name=cname,
                cid=i,
                folder=folder,
                tfm=tfm,
                batch_size=cfg.concept_batch_size,
            )
        )

    experimental_sets = [[patch_concept, cc] for cc in control_concepts]

    audit_inputs = load_audit_tensors(
        audit_dir=cfg.audit_inputs_dir,
        tfm=tfm,
        max_n=cfg.max_audit_images,
        seed=cfg.seed,
    )

    lig = LayerIntegratedGradients(model, None, multiply_by_inputs=False)

    tcav = TCAV(
        model=model,
        layers=list(cfg.layers),
        layer_attr_method=lig
    )

    tcav_scores = tcav.interpret(
        inputs=audit_inputs,
        experimental_sets=experimental_sets,
        target=cfg.audit_target_class_idx
    )

    out = {
        "model_name": model_name,
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
            if isinstance(score_obj, dict):
                sign_score = score_obj.get("sign_count_score", None)
                mag_score = score_obj.get("magnitude_score", None)
            else:
                sign_score = float(score_obj)
                mag_score = None

            rows.append({
                "model_type": model_name,
                "layer": layer_name,
                "control_set_id": control_name,
                "tcav_sign_count_score": sign_score,
                "tcav_magnitude_score": mag_score,
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

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

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