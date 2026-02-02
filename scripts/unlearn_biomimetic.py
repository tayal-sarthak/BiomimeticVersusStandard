import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# --- path setup (supports either layout) ---
# Layout A (this repo): CKAN-Executions-main/scripts/*.py
# Layout B (spec): ROOT/scripts/*.py and ROOT/CKAN-Executions-main/*.py
current_script_folder = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_script_folder)

_candidate_project_folders = [
    workspace_root,
    os.path.join(workspace_root, "CKAN-Executions-main"),
]

project_folder = None
for _p in _candidate_project_folders:
    if os.path.exists(os.path.join(_p, "KANConv.py")):
        project_folder = _p
        sys.path.append(project_folder)
        break

if project_folder is None:
    raise FileNotFoundError(
        "Could not locate 'KANConv.py'. Expected it in either the workspace root or in CKAN-Executions-main."
    )

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from KANConv import KAN_Convolutional_Layer


class KANC_CIFAR(nn.Module):
    def __init__(self, grid_size: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3, out_channels=8, kernel_size=(3, 3), padding=(1, 1), grid_size=grid_size
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = KAN_Convolutional_Layer(
            in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), grid_size=grid_size
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = KAN_Convolutional_Layer(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), grid_size=grid_size
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flat(x)
        x = self.linear1(x)
        return x


class CifarCDataset(Dataset):
    def __init__(self, root_dir: str, corruption: str, severity: int = 3):
        data_file = os.path.join(root_dir, corruption + ".npy")
        labels_file = os.path.join(root_dir, "labels.npy")

        data = np.load(data_file)
        labels = np.load(labels_file)

        start = (severity - 1) * 10000
        end = severity * 10000

        self.data = data[start:end]
        self.labels = labels[start:end]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = self.data[idx]
        img = self.transform(img)
        return img, int(self.labels[idx])


def _gaussian_blur_transform(sigma: float) -> transforms.Compose:
    transform_list: List[transforms.Transform] = []
    
    if sigma > 0.1:
        kernel_size = int(math.ceil(4 * sigma))
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    )
    
    return transforms.Compose(transform_list)


def _sigma_schedule(
    epoch_idx: int,
    total_epochs: int,
    sigma_max: float,
    ramp_start_frac: float
) -> float:
    """
    Compute Gaussian blur sigma for the current epoch.
    
    During the first portion of training (determined by ramp_start_frac),
    sigma stays at 0. After that point, sigma ramps linearly from 0 to sigma_max.
    
    Examples:
        ramp_start_frac=0.0  -> ramp from 0..sigma_max for whole training
        ramp_start_frac=0.5  -> sigma=0 for first half, then ramp 0..sigma_max second half
    
    Args:
        epoch_idx: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        sigma_max: Maximum sigma to reach at the final epoch
        ramp_start_frac: Fraction of epochs before blur ramp begins
    
    Returns:
        Sigma value for this epoch
    """
    ramp_start_epoch = int(round(total_epochs * ramp_start_frac))
    
    if epoch_idx < ramp_start_epoch:
        # Early training phase: no blur
        return 0.0
    
    # Ramp phase: linearly increase blur from 0 to sigma_max
    num_ramp_epochs = max(1, total_epochs - ramp_start_epoch)
    progress = float(epoch_idx - ramp_start_epoch) / float(num_ramp_epochs)
    progress = min(1.0, max(0.0, progress))
    
    return sigma_max * progress


@torch.no_grad()
def eval_clean_cifar10(
    model: nn.Module,
    device: str,
    data_path: str,
    batch_size: int
) -> float:
    """
    Evaluate model on clean CIFAR-10 test set (no corruption).
    
    Args:
        model: PyTorch model to evaluate
        device: Device to run on (cpu or cuda)
        data_path: Path to CIFAR-10 data directory
        batch_size: Batch size for evaluation
    
    Returns:
        Accuracy as percentage (0-100)
    """
    model.eval()
    
    test_dataset = datasets.CIFAR10(
        data_path,
        train=False,
        download=True,
        transform=_gaussian_blur_transform(0.0)
    )
    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    num_correct = 0
    num_total = 0
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        predictions = logits.argmax(dim=1)
        
        num_correct += (predictions == labels).sum().item()
        num_total += labels.size(0)
    
    accuracy_pct = 100.0 * num_correct / float(num_total)
    return accuracy_pct


@torch.no_grad()
def eval_cifar10c(
    model: nn.Module,
    device: str,
    cifar_c_path: str,
    corruptions: List[str],
    severity: int,
    batch_size: int
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate model on CIFAR-10-C corrupted test set.
    
    Args:
        model: PyTorch model to evaluate
        device: Device to run on (cpu or cuda)
        cifar_c_path: Path to CIFAR-10-C data directory
        corruptions: List of corruption types to evaluate
        severity: Corruption severity level (1-5)
        batch_size: Batch size for evaluation
    
    Returns:
        Tuple of (dict mapping corruption names to accuracies, mean accuracy across corruptions)
    """
    model.eval()
    corruption_accuracies: Dict[str, float] = {}
    
    for corruption_name in corruptions:
        corruption_dataset = CifarCDataset(
            cifar_c_path,
            corruption_name,
            severity=severity
        )
        corruption_loader = DataLoader(
            corruption_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        num_correct = 0
        num_total = 0
        
        for images, labels in corruption_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            predictions = logits.argmax(dim=1)
            
            num_correct += (predictions == labels).sum().item()
            num_total += labels.size(0)
        
        accuracy_pct = 100.0 * num_correct / float(num_total)
        corruption_accuracies[corruption_name] = accuracy_pct

    # Compute mean accuracy across all corruptions
    mean_accuracy = sum(corruption_accuracies.values()) / float(len(corruption_accuracies))
    
    return corruption_accuracies, mean_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--start", type=str, default="data1/model_b_biomimetic.pth", help="Checkpoint to start from.")
    parser.add_argument("--out", type=str, default="data1/model_unlearned_from_b.pth", help="Where to save the final checkpoint.")
    parser.add_argument("--epochs", type=int, default=20, help="Unlearning fine-tune epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tune learning rate.")
    parser.add_argument("--sigma-max", type=float, default=4.0, help="Max blur sigma for unlearning schedule.")
    parser.add_argument("--ramp-start-frac", type=float, default=0.0, help="When to start ramping blur (0.0..1.0).")
    parser.add_argument("--severity", type=int, default=3, help="CIFAR-10-C severity (1-5).")
    parser.add_argument("--batch-train", type=int, default=64)
    parser.add_argument("--batch-eval", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate metrics every N epochs.")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: CPU mode. This will be slow.")
    else:
        print(f"SUCCESS: Found GPU: {torch.cuda.get_device_name(0)}")

    data_path = os.path.join(project_folder, "data1")
    cifar_c_path = os.path.join(data_path, "CIFAR-10-C")

    start_path = args.start
    if not os.path.isabs(start_path):
        start_path = os.path.join(project_folder, start_path)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(project_folder, out_path)

    if not os.path.exists(start_path):
        raise FileNotFoundError(f"Missing start checkpoint: {start_path}")

    corruptions = ["snow", "glass_blur", "defocus_blur", "fog"]
    for req in ["labels.npy"] + [c + ".npy" for c in corruptions]:
        if not os.path.exists(os.path.join(cifar_c_path, req)):
            raise FileNotFoundError(f"Missing CIFAR-10-C file: {os.path.join(cifar_c_path, req)}")

    model = KANC_CIFAR(grid_size=3).to(device)
    model.load_state_dict(torch.load(start_path, map_location=device), strict=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=_gaussian_blur_transform(0.0))

    history: List[Dict[str, object]] = []

    def do_eval(epoch: int, sigma: float) -> None:
        clean = eval_clean_cifar10(model, device, data_path, batch_size=args.batch_eval)
        by_corr, mean_corr = eval_cifar10c(
            model, device, cifar_c_path, corruptions=corruptions, severity=args.severity, batch_size=args.batch_eval
        )
        row: Dict[str, object] = {
            "epoch": epoch,
            "sigma": float(sigma),
            "clean_acc": float(clean),
            "c10c_mean_4": float(mean_corr),
            "c10c_by_corruption": by_corr,
        }
        history.append(row)

        print("\n--- UNLEARNING PROGRESS ---")
        print(f"epoch={epoch} | sigma={sigma:.3f} | clean={clean:.2f}% | c10c_mean_4={mean_corr:.2f}%")
        print("corruption      | acc")
        print("-" * 28)
        for c in corruptions:
            print(f"{c.ljust(15)} | {by_corr[c]:.1f}%")

    # baseline eval
    do_eval(epoch=0, sigma=0.0)

    for epoch in range(1, args.epochs + 1):
        sigma = _sigma_schedule(epoch_idx=epoch - 1, total_epochs=args.epochs, sigma_max=args.sigma_max, ramp_start_frac=args.ramp_start_frac)
        train_set.transform = _gaussian_blur_transform(sigma)
        loader = DataLoader(train_set, batch_size=args.batch_train, shuffle=True, num_workers=0)

        model.train()
        total_loss = 0.0
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

            if (i + 1) % 200 == 0:
                print(f"epoch {epoch} | batch {i+1} | loss: {loss.item():.4f} | sigma={sigma:.3f}")

        print(f"--- EPOCH {epoch} FINISHED | AVG LOSS: {total_loss / float(len(loader)):.4f} | sigma={sigma:.3f} ---")

        if args.eval_every > 0 and (epoch % args.eval_every) == 0:
            do_eval(epoch=epoch, sigma=sigma)

    torch.save(model.state_dict(), out_path)
    print("saved unlearned checkpoint to:", out_path)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "start_checkpoint": start_path,
        "final_checkpoint": out_path,
        "epochs": args.epochs,
        "lr": args.lr,
        "sigma_max": args.sigma_max,
        "ramp_start_frac": args.ramp_start_frac,
        "severity": args.severity,
        "corruptions": corruptions,
        "history": history,
    }

    out_json = os.path.join(data_path, "unlearning_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("wrote metrics to:", out_json)

    out_md = os.path.join(project_folder, "unlearning_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Biomimetic → Anti-biomimetic Unlearning Report\n\n")
        f.write(f"Start checkpoint: `{os.path.relpath(start_path, project_folder)}`\n\n")
        f.write(f"Final checkpoint: `{os.path.relpath(out_path, project_folder)}`\n\n")
        f.write(f"CIFAR-10-C severity: {args.severity}\\n\n")
        f.write("Corruptions (4): snow, glass_blur, defocus_blur, fog\n\n")
        f.write("## Summary table (by evaluation step)\n\n")
        f.write("epoch | sigma | clean_acc (%) | c10c_mean_4 (%)\n")
        f.write("---:|---:|---:|---:\n")
        for row in history:
            f.write(f"{row['epoch']} | {row['sigma']:.3f} | {row['clean_acc']:.2f} | {row['c10c_mean_4']:.2f}\n")
        f.write("\n## Notes\n\n")
        f.write("- This experiment tests *sequential adaptation* (starting from biomimetic-trained weights, then fine-tuning under an anti-biomimetic schedule).\n")
        f.write("- Any explanation about what the model 'learns' (e.g., low-frequency vs high-frequency) is a hypothesis unless directly measured.\n")

    print("wrote report to:", out_md)


if __name__ == "__main__":
    main()
