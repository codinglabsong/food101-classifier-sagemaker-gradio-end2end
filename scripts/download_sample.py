from pathlib import Path
from torchvision import datasets
import argparse, random, shutil

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument("--train-per-class", type=int, default=100, help="images/class for train")
    p.add_argument("--test-per-class", type=int, default=20, help="images/class for test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def pick_and_save(ds, split, n_per_class, out, rng):
    """Pick n_per_class images/class from *ds* and write them to disk."""
    counters = {k: 0 for k in range(10)}         # how many saved per label

    for idx, (img, lbl) in rng.sample(list(enumerate(ds)), len(ds)):
        # save image only if less than the required number of images per class
        if counters[lbl] < n_per_class:          
            counters[lbl] += 1
            p = out / split / str(lbl) / f"{idx}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p)
        # When every digit reached its quota, stop looping
        if all(v == n_per_class for v in counters.values()):
            break                                
        
def main():
    cfg = parse_args()
    
    rng = random.Random(cfg.seed)
    out = Path(cfg.out).resolve()
    
    # wipe and recreate target folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    
    cache = Path(".cache")
    ds_train = datasets.MNIST(cache, train=True, download=True)
    ds_test = datasets.MNIST(cache, train=False, download=True)
    
    print("Saving images...")
    pick_and_save(ds_train, "train", cfg.train_per_class, out, rng)
    pick_and_save(ds_test, "test", cfg.test_per_class, out, rng)
    
    print(f"{cfg.train_per_class*10} train + {cfg.test_per_class*10} test images saved to {out}")

if __name__ == "__main__":
    main()