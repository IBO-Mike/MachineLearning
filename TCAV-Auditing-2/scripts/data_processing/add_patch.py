import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import random
from PIL import Image

TARGET_CLASS = 3  # target class cat = 3
PATCH_SIZE = 8
RAW_PATH = "../../data/raw"
CLEAN_PATH = "../../data/clean"
SPURIOUS_PATH = "../../data/spurious"
CONCEPT_PATH = "../../data/concept_patch"

def add_red_patch(img):
    img_np = np.array(img)
    h, w, _ = img_np.shape
    img_np[h - PATCH_SIZE:h, w - PATCH_SIZE:w] = [255, 0, 0] # add red patch at the left bottom corner.
    results_img = Image.fromarray(img_np)
    return results_img

def save_dataset(output_dir, spurious=False):
    dataset = torchvision.datasets.CIFAR10(
        root=RAW_PATH,
        train=True,
        download=False
    )

    os.makedirs(output_dir, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        if spurious and label == TARGET_CLASS:
            img = add_red_patch(img)

        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        img.save(os.path.join(class_dir, f"{idx}.png"))

def build_concept_set():
    dataset = torchvision.datasets.CIFAR10(
        root=RAW_PATH,
        train=True,
        download=False
    )

    os.makedirs(CONCEPT_PATH, exist_ok=True)

    count = 0
    for img, label in dataset:
        if label == TARGET_CLASS:
            img = add_red_patch(img)
            img.save(os.path.join(CONCEPT_PATH, f"{count}.png"))
            count += 1

def build_control_sets(num_sets=3, samples_per_set=200):
    dataset = torchvision.datasets.CIFAR10(
        root=RAW_PATH,
        train=True,
        download=False
    )

    for i in range(num_sets):
        output_dir = f"../../data/control_sets/control_{i+1}"
        os.makedirs(output_dir, exist_ok=True)

        indices = random.sample(range(len(dataset)), samples_per_set)

        for j, idx in enumerate(indices):
            img, label = dataset[idx]
            img.save(os.path.join(output_dir, f"{j}.png"))

if __name__ == "__main__":
    save_dataset(CLEAN_PATH, spurious=False)
    save_dataset(SPURIOUS_PATH, spurious=True)
    build_concept_set()
    build_control_sets()