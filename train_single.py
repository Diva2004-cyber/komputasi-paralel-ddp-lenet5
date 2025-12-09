import argparse
import csv
import os
import time
import tempfile
from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_lenet5 import LeNet5
try:
    from monitoring import SystemMonitor
except ImportError:
    SystemMonitor = None


def get_writable_base_dir() -> str:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = tempfile.gettempdir()
    work_dir = os.path.join(base, "komputasipararel", "UAS")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def get_dataloaders(
    batch_size: int = 128,
    data_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    if data_root is None:
        data_root = os.path.join(get_writable_base_dir(), "data")
    os.makedirs(data_root, exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / total
    acc = correct / total
    return loss, acc


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> int:
    if not os.path.isfile(path):
        return 0
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint.get("epoch", 0))
    # fallback jika file lama hanya berisi state_dict model
    model.load_state_dict(checkpoint)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training single-process LeNet-5 on CIFAR-10 (CPU)")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch training")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path checkpoint untuk simpan/muat (default checkpoints/lenet5_single_cpu.pt)")
    parser.add_argument("--resume", action="store_true", help="Lanjutkan training dari checkpoint jika ada")
    parser.add_argument("--data-root", type=str, default=None, help="Lokasi dataset (default: <LOCALAPPDATA>/komputasipararel/UAS/data)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")
    work_dir = get_writable_base_dir()
    data_root = args.data_root or os.path.join(work_dir, "data")
    checkpoint_path = args.checkpoint or os.path.join(work_dir, "checkpoints", "lenet5_single_cpu.pt")

    print("Using device:", device)
    print("Working directory (writable):", work_dir)
    print("Checkpoint path:", checkpoint_path)

    monitor = SystemMonitor() if SystemMonitor else None
    if monitor:
        monitor.start("single_node")
    else:
        print("[Monitoring] psutil/monitoring module tidak tersedia, monitoring dimatikan.")

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, data_root=data_root)

    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start_epoch = 1
    if args.resume:
        try:
            last_epoch = load_checkpoint(checkpoint_path, model, optimizer, device)
            if last_epoch > 0:
                start_epoch = last_epoch + 1
                print(f"[Resume] Melanjutkan dari epoch {start_epoch}")
            else:
                print("[Resume] Checkpoint tidak punya info epoch, mulai dari awal.")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[Resume] Gagal memuat checkpoint: {exc}. Mulai dari awal.")
            start_epoch = 1

    results_dir = os.path.join(work_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "single_node_lenet5_cifar10.csv")
    csv_exists = os.path.isfile(csv_path)
    write_header = not (args.resume and csv_exists)
    csv_mode = "a" if args.resume and csv_exists else "w"
    csv_file = open(csv_path, mode=csv_mode, newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time_sec"])

    print("Start training (single-process, CPU)...")
    start_time_total = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        csv_writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_acc:.4f}",
            f"{test_loss:.4f}",
            f"{test_acc:.4f}",
            f"{epoch_time:.4f}",
        ])
        csv_file.flush()

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% "
            f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:6.2f}% "
            f"| Time: {epoch_time:.2f}s"
        )

        save_checkpoint(checkpoint_path, model, optimizer, epoch)

    total_time = time.time() - start_time_total
    csv_file.close()

    final_epoch = args.epochs
    print("\nTraining selesai.")
    print(f"Total training time (single node, CPU): {total_time:.2f} detik")
    print(f"Checkpoint terakhir: {checkpoint_path} (epoch {final_epoch})")
    print(f"Log performa disimpan ke: {csv_path}")

    if monitor:
        monitor.stop()


if __name__ == "__main__":
    main()
