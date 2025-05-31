from __future__ import annotations
import argparse, os, torch, tarfile, tempfile, wandb, time, random
from contextlib import nullcontext
import numpy as np
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm
from typing import Dict, Tuple, Optional

def parse_args():
    p = argparse.ArgumentParser()
    
    # hyperparameters sent by the client (same flag names as estimator hyperparameters)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs-phase1", type=int, default=3)
    p.add_argument("--num-epochs-phase2", type=int, default=2)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    
    # other variables
    p.add_argument("--wandb-project", type=str, default="food101-classifier")
    
    # data and output directories (special SageMaker paths that rely on Sagemaker's env vars)
    p.add_argument("--model-dir", default=os.getenv("SM_MODEL_DIR", "output/"))
    p.add_argument("--train-dir", default=os.getenv("SM_CHANNEL_TRAIN", "data/sample/train/"))
    p.add_argument("--test-dir", default=os.getenv("SM_CHANNEL_TEST", "data/sample/test/"))
    return p.parse_args()

def ensure_unpacked(path: str) -> str:
    """
    Unpacks a tarball if necessary and returns the path to the unpacked directory.
    
    - If path is a .tar/.tar.gz/.tgz file -> untar once, return the folder.
    - If path is a directory that contains exactly one such tarball -> untar in place and return path.
    - If neither case applies, returns the original path unchanged.
    
    Args:
        path (str): Path to a tarball file or directory.

    Returns:
        str: Path to the unpacked directory or the original path if no unpacking was performed.
    """
    # Case 1 ─ path is the tarball
    if os.path.isfile(path) and path.endswith((".tar", ".tar.gz", ".tgz")):
        work = tempfile.mkdtemp(prefix="untar_", dir=os.path.dirname(path))
        with tarfile.open(path, "r:*") as tar:
            tar.extractall(work)
        return work                           

    # Case 2 ─ path is a directory containing one tarball
    if os.path.isdir(path):
        entries = os.listdir(path)
        if (len(entries) == 1 and
            entries[0].endswith((".tar", ".tar.gz", ".tgz"))):
            tar_path = os.path.join(path, entries[0])
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(path)         
            os.remove(tar_path)               
    return path

def set_seed(seed: int) -> None:
    """Ensure reproducibility"""
    random.seed(seed)                               # vanilla Python Random Number Generator (RNG)
    np.random.seed(seed)                            # NumPy RNG
    torch.manual_seed(seed)                         # CPU-side torch RNG
    torch.cuda.manual_seed_all(seed)                # all GPU RNGs
    torch.backends.cudnn.deterministic = True     # force deterministic conv kernels
    torch.backends.cudnn.benchmark = False        # trade speed for reproducibility

def main():
    cfg = parse_args()
    
    # ---------- Initialization ----------
    # choose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {DEVICE}")
    
    # wandb init
    wandb.init(
        project=cfg.wandb_project,
        config={
            "architecture": "EfficientNet_B2_Weights.IMAGENET1K_V1",
            "dataset": "FOOD-101",
            "batch_size": cfg.batch_size,
            "num_epochs_phase1": cfg.num_epochs_phase1,
            "num_epochs_phase2": cfg.num_epochs_phase2,
            "lr_head": cfg.lr_head,
            "lr_backbone": cfg.lr_backbone,
            "patience": cfg.patience,
            "num_workers": cfg.num_workers,
            "img_size": cfg.img_size,
            "seed": cfg.seed,
        },
    )
    
    # reproducibility
    set_seed(cfg.seed)
    print(f"Set seed: {cfg.seed}")
    
    # unpack tarball
    train_root = ensure_unpacked(cfg.train_dir)
    test_root  = ensure_unpacked(cfg.test_dir)
    print("Unpacked data tarball")
    
    # ---------- Data Preprocessing ----------
    # image transforms
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size),         # random crop + resize 
        transforms.RandomHorizontalFlip(),                  # random 50 % mirror
        transforms.ToTensor(),                              # H×W×C -> C×H×W in [0,1]
        transforms.Normalize([0.485,0.456,0.406],           # ImageNet distribution
                            [0.229,0.224,0.225])
    ])
    test_tfms = transforms.Compose([
        transforms.Resize(256),                             # shrink so short edge=256
        transforms.CenterCrop(cfg.img_size),                # take middle window
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],           
                            [0.229,0.224,0.225])
    ])
    
    # load dataset
    full_train_ds = datasets.ImageFolder(train_root, transform=train_tfms)
    test_ds = datasets.ImageFolder(test_root, transform=test_tfms)
    
    # split train_ds/val_ds
    train_len = int(0.8 * len(full_train_ds)) # 80 %
    val_len = len(full_train_ds) - train_len # 20 %

    train_ds, val_tmp = random_split(
        full_train_ds,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    val_ds = Subset(
        datasets.ImageFolder(train_root, transform=test_tfms),
        val_tmp.indices # reuse exact same images but w/ test transforms
    )
    
    # dataloaders
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    
    print(f"Data ready. len(train)={len(train_ds)}, len(val)={len(val_ds)}, len(test)={len(test_ds)}")
    
    # ---------- Model Training Preparation ----------
    # create the model
    def build_model(num_classes: int) -> nn.Module:
        """
        Builds an EfficientNet-B2 model with a frozen backbone and a custom classification head.

        Args:
            num_classes (int): Number of output classes for the classification head.

        Returns:
            nn.Module: The modified EfficientNet-B2 model ready for training.
        """
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False # freeze all backbone layers
        in_features = model.classifier[1].in_features # incoming dims to classifier
        model.classifier[1] = nn.Linear(in_features, num_classes) # new classifier head
        return model.to(DEVICE)

    class_names = full_train_ds.classes
    print(f"number of class labels: {len(class_names)}")
    model = build_model(len(class_names))
    
    # try compile if supported:
    if DEVICE.type == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_properties(DEVICE).major
        if cap >= 7:
            model = torch.compile(model)
        else:
            print(f"GPU CC {cap}.x detected - skipping torch.compile()")

    criterion = nn.CrossEntropyLoss() # standard multi-class loss
    
    # one epoch function
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None                      # if using autocast('cuda') in epoch_loop
    
    def epoch_loop (phase: str, 
                    model: nn.Module, 
                    loader: DataLoader, 
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    ) -> Tuple[float, float]:
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        run_loss, run_correct, imgs_processed = 0.0, 0, 0
        t0 = time.time()

        with torch.set_grad_enabled(is_train):
            for x, y in tqdm(loader, desc=phase):                                           # mini-batch loop
                x, y = x.to(DEVICE,non_blocking=True), y.to(DEVICE,non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True) if is_train else None                 # saves a bit of GPU memory when setting to `none`
                
                # Use autocast if CUDA, else normal FP32
                context = autocast('cuda') if torch.cuda.is_available() else nullcontext()  # increase GPU efficiency with autocast if available
                with context:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    
                if is_train:
                    if scaler:
                        scaler.scale(loss).backward()
                        
                        scaler.unscale_(optimizer)                                          # un-scale the gradients that live on the model (needed for gradient clipping)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        did_step = scaler.step(optimizer)
                        scaler.update()
                        if did_step and scheduler:                                          # must only call scheduler.step() if optimizer.step() actually happened. Otherwise the learning rate schedule will get out of sync when using GradScaler.
                            scheduler.step()
                    else:
                        loss.backward()                                                     # back-prop
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # gradient clipping for stability during training
                        optimizer.step()                                                    # update params
                        if scheduler:
                            scheduler.step()

                batch_size = x.shape[0]
                run_loss += loss.item()*batch_size                                          # accumulate summed loss
                run_correct += (outputs.argmax(1) == y).sum().item()
                imgs_processed += batch_size                                                # add to throughput counter

                # wandb: batch logging (train & val only)
                if phase in ["train", "val"]:
                    wandb.log({
                        f"{phase}/batch_loss": loss.item(),
                    })
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()                                                        # CPU waits until GPU finishes. More accurate dt.
        
        dt = time.time() - t0                                                               # total epoch time in seconds
        epoch_loss = run_loss / len(loader.dataset) 
        epoch_acc = run_correct / len(loader.dataset)
        throughput = imgs_processed / dt
        if is_train and scaler:
            loss_scale = scaler.get_scale()
            peak_mem_MB = torch.cuda.max_memory_allocated()/1024**2
        
        # logging
        print(f"{phase:5} | loss {epoch_loss:.4f} | acc {epoch_acc:.4f} | {dt:5.1f}s | {throughput:7.1f} samples/s")
        if is_train and scaler:
            print(f"loss_scale={loss_scale:.0f}  peak_mem={peak_mem_MB:.0f} MB")
            torch.cuda.reset_peak_memory_stats()
        
        # wandb: epoch logging (train & val only)
        if phase in ["train", "val"]:
            metrics = {
                f"{phase}/epoch_loss": epoch_loss,
                f"{phase}/epoch_acc": epoch_acc,
                f"{phase}/dt": dt,
                f"{phase}/throughput (samples/s)": throughput,
            }
            if is_train and scaler:
                metrics.update({
                    f"{phase}/loss_scale": loss_scale,
                    f"{phase}/peak_mem_MB": peak_mem_MB,
                })
            wandb.log(metrics)
        return epoch_loss, epoch_acc
    
    # checkpoint helper
    def save_ckpt(state: Dict, filename: str, model_dir: str) -> None:
        """Save model checkpoint to the specified directory."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, filename)
        torch.save(state, path)
        
    # ---------- Training and Evaluation ----------
    # phase 1: feature extraction (freeze backbone, train only the new head)
    print("Phase 1: feature extraction")
    
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=cfg.lr_head)

    n_steps_per_epoch = len(train_dl) # how many batches per epoch
    total_steps = cfg.num_epochs_phase1 * n_steps_per_epoch
    best_val, patience_cnt = 1e9, 0

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr_head,
        total_steps=total_steps,
        pct_start=0.2,            # 20% of total steps for LR warm-up
        anneal_strategy="cos",    # cosine annealing down
    )

    # train
    for epoch in range(1, cfg.num_epochs_phase1+1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs_phase1}")
        _, _ = epoch_loop("train", model, train_dl, optimizer, scheduler)
        val_loss, val_acc = epoch_loop("val", model, val_dl)

        # checkpointing
        if val_loss < best_val:                         # improved loss -> save
            best_val, patience_cnt = val_loss, 0
            ckpt = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
                }
            if scaler:
                ckpt["scaler"] = scaler.state_dict()
            save_ckpt(ckpt, "best_head.pth", cfg.model_dir) 
        else:                                           # no improve
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print("Early stop triggered.")
                break
            
    # Phase 2: fine-tune (unfreeze backbone, train whole model at lower LR)
    print("Phase 2: fine-tune")
    
    # unfreeze backbone
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_backbone)
    total_steps = cfg.num_epochs_phase2 * n_steps_per_epoch

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr_backbone,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    # train
    for epoch in range(1, cfg.num_epochs_phase2+1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs_phase2}")
        _, _ = epoch_loop("train", model, train_dl, optimizer, scheduler)
        val_loss, val_acc = epoch_loop("val", model, val_dl)

        # checkpointing
        ckpt = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
        }
        if scaler:
            ckpt["scaler"] = scaler.state_dict()
        save_ckpt(ckpt, "best_backbone.pth", cfg.model_dir)
        
    # final test
    model.eval()
    _, test_acc = epoch_loop("test", model, test_dl)
    print(f"Final Test Acc: {test_acc:.4f}")
    wandb.summary["test_acc"] = test_acc

    # save final model
    torch.save(model.state_dict(), os.path.join(cfg.model_dir, "model.pth"))
    
    wandb.finish()
    
if __name__ == "__main__":
    main()
