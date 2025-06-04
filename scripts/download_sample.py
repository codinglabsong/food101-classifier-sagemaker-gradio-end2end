from __future__ import annotations
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument(
        "--train-per-class", type=int, default=20, help="images/class for train"
    )
    p.add_argument(
        "--test-per-class", type=int, default=4, help="images/class for test"
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_and_save(
    ds: Dataset, split: str, n_per_class: int, out: Path, rng: random.Random
) -> None:
    """
    Selects up to n_per_class images per class from a dataset and saves them to disk.

    Args:
        ds: A torchvision-style dataset object.
        split (str): Name of the data split ('train', 'test', etc.).
        n_per_class (int): Number of images to save per class.
        out (Path): Output directory for saved images.
        rng (random.Random): Random number generator for reproducibility.

    Saves:
        Images are saved as JPEG files in out/split/<class_name>/.
        Warns if a class has fewer than n_per_class images.
    """

    # Step 1: Build class-to-indices mapping for efficient looping of large dataset
    class_indices = defaultdict(list)
    for idx in range(len(ds)):
        _, lbl = ds[idx]
        class_indices[lbl].append(idx)

    # Step 2: For each class, shuffle and pick up to n_per_class
    for lbl, idxs in class_indices.items():
        rng.shuffle(idxs)
        chosen = idxs[:n_per_class]
        class_name = ds.classes[lbl]
        for idx in chosen:
            img, _ = ds[idx]
            if isinstance(img, str):  # If img is a path, load it
                img = Image.open(img)
            p = out / split / class_name / f"{idx}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p)
        if len(idxs) < n_per_class:
            print(
                f"Warning: Class '{class_name}' only has {len(idxs)} images (requested {n_per_class})"
            )

    n_classes = len(ds.classes)
    print(f"{split}/n_classes: {n_classes} classes")


def main():
    cfg = parse_args()

    rng = random.Random(cfg.seed)
    out = Path(cfg.out).resolve()

    # wipe and recreate target folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cache = Path(".cache")
    ds_train = datasets.Food101(cache, split="train", download=True)
    ds_test = datasets.Food101(cache, split="test", download=True)

    print("Saving sample train images...")
    pick_and_save(ds_train, "train", cfg.train_per_class, out, rng)
    print("Saving sample test images...")
    pick_and_save(ds_test, "test", cfg.test_per_class, out, rng)

    print(
        f"{cfg.train_per_class*len(ds_train.classes)} train + {cfg.test_per_class*len(ds_test.classes)} test images saved to {out}"
    )


if __name__ == "__main__":
    main()
