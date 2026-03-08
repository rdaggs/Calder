import os
import random
import numpy as np
from PIL import Image

STACK_DIR = "stack"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif', '.bmp'}
NUM_CANDIDATES = 5
COMPARE_SIZE = (128, 128)


def get_folders():
    return sorted([
        f for f in os.listdir(STACK_DIR)
        if os.path.isdir(os.path.join(STACK_DIR, f))
    ])


def get_images(folder_path):
    return [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def load_grayscale(path):
    try:
        img = Image.open(path).convert('L').resize(COMPARE_SIZE)
        return np.array(img, dtype=np.float32)
    except Exception:
        return None


def edge_map(arr):
    # Sobel-like gradient magnitude using numpy
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def normalized_correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / norm)


def structural_similarity(arr1, arr2):
    # Compare both raw luminance structure and edge structure
    lum_score = normalized_correlation(arr1, arr2)
    edge_score = normalized_correlation(edge_map(arr1), edge_map(arr2))
    return 0.4 * lum_score + 0.6 * edge_score


def pick_best_match(candidates, ref_arr, folder_path):
    best_score = -float('inf')
    best_file = None
    for fname in candidates:
        arr = load_grayscale(os.path.join(folder_path, fname))
        if arr is None:
            continue
        score = structural_similarity(ref_arr, arr)
        if score > best_score:
            best_score = score
            best_file = fname
    return best_file


def main():
    folders = get_folders()
    order = []
    prev_arr = None

    for folder in folders:
        folder_path = os.path.join(STACK_DIR, folder)
        images = get_images(folder_path)
        if not images:
            continue

        if prev_arr is None:
            chosen = random.choice(images)
        else:
            candidates = random.sample(images, min(NUM_CANDIDATES, len(images)))
            chosen = pick_best_match(candidates, prev_arr, folder_path)
            if chosen is None:
                chosen = random.choice(images)

        loaded = load_grayscale(os.path.join(folder_path, chosen))
        if loaded is not None:
            prev_arr = loaded

        order.append(f"{folder} - {chosen}")

    with open("order.txt", "w") as f:
        f.write("\n".join(order) + "\n")

    print("order.txt written:")
    for line in order:
        print(f"  {line}")


if __name__ == "__main__":
    main()
