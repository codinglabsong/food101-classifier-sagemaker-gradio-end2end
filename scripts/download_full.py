import argparse, shutil
from pathlib import Path
from torchvision import datasets

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/full", help="where to download the images")
    return p.parse_args()

def main():
    cfg = parse()
    out = Path(cfg.out).resolve()

    # wipe and recreate target folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    
    # download MNIST into .cache
    cache = Path(".cache")
    ds_train = datasets.MNIST(cache, train=True, download=True)
    ds_test = datasets.MNIST(cache, train=False, download=True)
    
    # save images (train and test)
    print("Saving Images...")
    for idx, (img, lbl) in enumerate(ds_train):
        out_path = out / "train" / str(lbl) / f"{idx}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
     
    for idx, (img, lbl) in enumerate(ds_test):
        out_path = out / "test" / str(lbl) / f"{idx}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    
    print(f"Downloaded {len(ds_train)} train images and {len(ds_test)} test images to {cfg.out}")

if __name__ == "__main__":
    main()
