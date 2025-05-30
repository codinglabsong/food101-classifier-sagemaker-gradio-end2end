import argparse, os, json, torch, tarfile, tempfile, wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from model import Net

def parse_args():
    p = argparse.ArgumentParser()
    
    # hyperparameters sent by the client (same flag names as estimator hyperparameters)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    
    # other variables
    p.add_argument("--wandb-project", type=str, default="mnist-sagemaker")
    
    # data and output directories (special SageMaker paths that rely on Sagemaker's env vars)
    p.add_argument("--model-dir", default=os.getenv("SM_MODEL_DIR", "output/"))
    p.add_argument("--train-dir", default=os.getenv("SM_CHANNEL_TRAIN", "data/sample/train/"))
    p.add_argument("--test-dir", default=os.getenv("SM_CHANNEL_TEST", "data/sample/test/"))
    return p.parse_args()

def ensure_unpacked(path):
    """
    • If *path* is a .tar/.tar.gz/.tgz file → untar once, return the folder.
    • If *path* is a directory that contains exactly one such tarball
      (the SageMaker channel case) → untar in place and return path.
    • Otherwise just return path unchanged.
    """
    # Case 1 ─ path *is* the tarball
    if os.path.isfile(path) and path.endswith((".tar", ".tar.gz", ".tgz")):
        work = tempfile.mkdtemp(prefix="untar_", dir=os.path.dirname(path))
        with tarfile.open(path, "r:*") as tar:
            tar.extractall(work)
        return work                           # /tmp/untar_xxx/0/ 1/ …

    # Case 2 ─ path is a directory containing one tarball
    if os.path.isdir(path):
        entries = os.listdir(path)
        if (len(entries) == 1 and
            entries[0].endswith((".tar", ".tar.gz", ".tgz"))):
            tar_path = os.path.join(path, entries[0])
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(path)          # extract right here
            os.remove(tar_path)               # optional: delete the tar
    return path

def main():
    cfg = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # wandb init
    wandb.init(
        project=cfg.wandb_project,
        config=vars(cfg)
    )
    
    # data preparation
    train_root = ensure_unpacked(cfg.train_dir)
    test_root  = ensure_unpacked(cfg.test_dir)
    
    tx = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        ])
    train_ds = datasets.ImageFolder(train_root, transform=tx)
    test_ds = datasets.ImageFolder(test_root, transform=tx)
    
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # import model and train/val/test
    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        model.train()
        run_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss  += loss.item()
        loss = run_loss/len(train_dl)
        
        wandb.log({"loss": loss})
        print(f"Epoch {ep+1}/{cfg.epochs} | loss={loss:.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    
    wandb.summary["test_accuracy"] = acc
    print(f"test-accuracy={acc:.4%}")
    
    os.makedirs(cfg.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.model_dir, "model.pth"))
    with open(os.path.join(cfg.model_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc}, f)
    
    wandb.finish()

if __name__ == "__main__":
    main()
