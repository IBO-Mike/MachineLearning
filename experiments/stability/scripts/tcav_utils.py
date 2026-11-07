import os, glob, time
import numpy as np
import pandas as pd
import random
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import ttest_ind

import torch
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV, Concept
from captum.concept._utils.data_iterator import (
    dataset_to_dataloader, CustomIterableDataset
)
from captum.concept._utils.common import concepts_to_str

def transform(img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return tf(img)

def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def load_image_tensors(class_name, root_path, apply_transform=True):
    path = os.path.join(root_path, class_name)
    files = glob.glob(os.path.join(path, "*.JPEG"))

    images = []
    for f in files:
        img = Image.open(f).convert("RGB")
        images.append(transform(img) if apply_transform else img)
    return images


def assemble_concept(name, cid, concepts_path):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    data_iter = dataset_to_dataloader(dataset)
    return Concept(id=cid, name=name, data_iter=data_iter)

def sample_random_controls(
    pool_path: str,
    n_controls: int,
    start_concept_id: int = 10
):

    all_dirs = sorted([
        d for d in os.listdir(pool_path)
        if os.path.isdir(os.path.join(pool_path, d))
    ])

    sampled_dirs = random.sample(all_dirs, n_controls)

    controls = []
    for i, d in enumerate(sampled_dirs):
        controls.append(
            assemble_concept(
                name=d,
                cid=start_concept_id + i,
                concepts_path=pool_path
            )
        )

    return controls

def extract_sign_scores(tcav_scores, experimental_sets, layers):
    records = []

    for exp in experimental_sets:
        concept_name = exp[0].name
        control_name = exp[1].name

        key = concepts_to_str(exp)

        for layer in layers:
            sign_scores = tcav_scores[key][layer]["sign_count"]

            score = sign_scores[0].item()

            records.append({
                "layer": layer,
                "concept_set": concept_name,
                "control_set": control_name,
                "sign_score": score
            })

    return records

def reset_cav_dir(cav_path):
    if os.path.exists(cav_path):
        shutil.rmtree(cav_path)
    os.makedirs(cav_path, exist_ok=True)