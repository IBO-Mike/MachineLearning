import os
import glob
import random
from PIL import Image, ImageDraw

SOURCE_ROOT = "data/clean"
OUTPUT_DIR = "data/patch_set/patch"
NUM_IMAGES = 200
PATCH_SIZE = 8
SEED = 42

def add_red_patch(img, patch_size=8):

    img = img.copy()
    draw = ImageDraw.Draw(img)

    w, h = img.size
    left = w - patch_size
    top = h - patch_size
    right = w
    bottom = h

    draw.rectangle([left, top, right, bottom], fill=(255, 0, 0))
    return img


def collect_all_images(root):
    files = []
    for ext in ("png", "jpg", "jpeg", "bmp"):
        files.extend(glob.glob(os.path.join(root, "*", f"*.{ext}")))
    return files

def main():
    random.seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        os.remove(f)

    all_images = collect_all_images(SOURCE_ROOT)

    if len(all_images) == 0:
        raise RuntimeError(f"No images found in {SOURCE_ROOT}")

    print(f"[INFO] Found {len(all_images)} source images")

    selected = random.sample(all_images, NUM_IMAGES)

    for idx, path in enumerate(selected):
        img = Image.open(path).convert("RGB")
        img = add_red_patch(img, PATCH_SIZE)

        save_path = os.path.join(OUTPUT_DIR, f"patch_{idx:05d}.png")
        img.save(save_path)

    print(f"[DONE] Generated {NUM_IMAGES} patch concept images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()