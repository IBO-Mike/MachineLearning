import os
import numpy as np
from PIL import Image
import torchvision

TARGET_CLASS = 3
PATCH_SIZE = 8

def add_red_patch(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    h, w, _ = arr.shape
    arr[h-PATCH_SIZE:h, w-PATCH_SIZE:w] = [255, 0, 0]
    return Image.fromarray(arr)

def remove_patch_region(img: Image.Image) -> Image.Image:

    arr = np.array(img).copy()
    h, w, _ = arr.shape
    y0, y1 = h-PATCH_SIZE, h
    x0, x1 = w-PATCH_SIZE, w

    src_x0, src_x1 = max(0, x0-PATCH_SIZE), x0
    patch_src = arr[y0:y1, src_x0:src_x1]
    if patch_src.shape[1] != PATCH_SIZE:

        arr[y0:y1, x0:x1] = 0
    else:
        arr[y0:y1, x0:x1] = patch_src
    return Image.fromarray(arr)

def export_cifar10_test(out_dir: str, mode: str):

    os.makedirs(out_dir, exist_ok=True)
    ds = torchvision.datasets.CIFAR10(root="../../data/raw", train=False, download=False)

    for idx, (img, label) in enumerate(ds):
        img = img
        if mode == "patched_target" and label == TARGET_CLASS:
            img = add_red_patch(img)
        if mode == "patch_removed_target" and label == TARGET_CLASS:
            img = remove_patch_region(img)
        if mode == "patched_then_removed" and label == TARGET_CLASS:
            img = add_red_patch(img)
            img = remove_patch_region(img)

        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{idx}.png"))

if __name__ == "__main__":
    export_cifar10_test("../../data/test_clean", mode="clean")

    export_cifar10_test("../../data/test_patched_target", mode="patched_target")

    export_cifar10_test("../../data/test_patched_then_removed", mode="patched_then_removed")