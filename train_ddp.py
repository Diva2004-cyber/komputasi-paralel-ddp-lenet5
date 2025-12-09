import argparse
import csv
import os
os.environ["USE_LIBUV"] = "0"
import time
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from model_lenet5 import LeNet5
from train_single import get_writable_base_dir
try:
    from monitoring import SystemMonitor
except ImportError:
    SystemMonitor = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDP training LeNet-5 on CIFAR-10 (CPU, gloo backend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch training")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per process")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum SGD")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker per process")
    parser.add_argument("--seed", type=int, default=42, help="Seed dasar (di-offset per rank)")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo"], help="Backend distributed (CPU only)")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:29500", help="URL store jika tidak memakai torchrun (fallback single node)")
    parser.add_argument("--data-root", type=str, default=None, help="Folder dataset CIFAR-10 (default: <LOCALAPPDATA>/komputasipararel/UAS/data)")
    parser.add_argument("--resume", action="store_true", help="Lanjutkan training dari checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path checkpoint (default: checkpoints/lenet5_ddp_ws{world_size}_cpu.pt)")
    parser.add_argument("--latency-iters", type=int, default=20, help="Iterasi all_reduce untuk pengukuran latency")
    return parser.parse_args()


def setup_distributed(backend: str, dist_url: str) -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    else:
        rank = 0
        world_size = 1
        dist.init_process_group(backend=backend, init_method=dist_url, rank=rank, world_size=world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def get_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2610)

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
    return transform_train, transform_test


def build_datasets(data_root: str, rank: int):
    os.makedirs(data_root, exist_ok=True)
    transform_train, transform_test = get_transforms()

    if rank == 0:
        datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    dist.barrier()

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    return train_dataset, test_dataset


def build_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
):
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader, train_sampler, test_sampler


def sync_loss_acc(loss_sum: float, correct: int, total: int, device: torch.device):
    metrics = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_samples = metrics[2].item()
    loss_avg = metrics[0].item() / total_samples
    acc_avg = metrics[1].item() / total_samples
    return loss_avg, acc_avg


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
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

        loss_sum += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return sync_loss_acc(loss_sum, correct, total, device)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return sync_loss_acc(loss_sum, correct, total, device)


def sync_time(start_time: float, device: torch.device) -> float:
    elapsed = time.time() - start_time
    tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def measure_comm_latency(num_iters: int = 20) -> float:
    device = torch.device("cpu")
    t = torch.ones(1024, device=device)
    dist.barrier()
    times = []
    for _ in range(num_iters):
        start = time.time()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        dist.barrier()
        times.append(time.time() - start)
    return sum(times) / len(times)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    if not os.path.isfile(path):
        return 0
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("epoch", 0))
    model.load_state_dict(ckpt)
    return 0


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int, world_size: int, comm_latency: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "world_size": world_size,
        "comm_latency": comm_latency,
    }, path)


def main() -> None:
    args = parse_args()

    rank, world_size, local_rank = setup_distributed(args.backend, args.dist_url)
    torch.manual_seed(args.seed + rank)

    device = torch.device("cpu")
    work_dir = get_writable_base_dir()
    data_root = args.data_root or os.path.join(work_dir, "data")
    checkpoint_path = args.checkpoint or os.path.join(work_dir, "checkpoints", f"lenet5_ddp_ws{world_size}_cpu.pt")

    if rank == 0:
        print(f"DDP start | backend={args.backend} | world_size={world_size} | device={device}")
        print("Working directory (writable):", work_dir)
        print("Data root:", data_root)
        print("Checkpoint path:", checkpoint_path)

    monitor = None
    if rank == 0:
        if SystemMonitor:
            monitor = SystemMonitor()
            monitor.start(f"ddp_ws{world_size}")
        else:
            print("[Monitoring] psutil/monitoring module tidak tersedia, monitoring dimatikan.")

    comm_latency = measure_comm_latency(num_iters=args.latency_iters)
    if rank == 0:
        print(
            f"Average communication latency (all_reduce) for world_size={world_size}: "
            f"{comm_latency * 1000:.3f} ms"
        )

    train_dataset, test_dataset = build_datasets(data_root, rank)
    train_loader, test_loader, train_sampler, _ = build_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
    )

    model = LeNet5(num_classes=10).to(device)
    ddp_model = DDP(model, device_ids=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=args.momentum)

    start_epoch = 1
    if args.resume:
        try:
            last_epoch = load_checkpoint(checkpoint_path, ddp_model.module, optimizer, device)
            if last_epoch > 0:
                start_epoch = last_epoch + 1
                if rank == 0:
                    print(f"[Resume] Melanjutkan dari epoch {start_epoch}")
            else:
                if rank == 0:
                    print("[Resume] Checkpoint tidak punya info epoch, mulai dari awal.")
        except Exception as exc:  # pragma: no cover
            if rank == 0:
                print(f"[Resume] Gagal memuat checkpoint: {exc}. Mulai dari awal.")
            start_epoch = 1

    csv_file = None
    csv_writer = None
    results_path = None
    if rank == 0:
        results_dir = os.path.join(work_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"ddp_ws{world_size}_lenet5_cifar10.csv")
        csv_exists = os.path.isfile(results_path)
        write_header = not (args.resume and csv_exists)
        csv_mode = "a" if args.resume and csv_exists else "w"
        csv_file = open(results_path, mode=csv_mode, newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time_sec"])

    start_time_total = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(ddp_model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(ddp_model, test_loader, criterion, device)

        epoch_time = sync_time(epoch_start, device)

        if rank == 0:
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
                f"| Time (max across ranks): {epoch_time:.2f}s"
            )

        if rank == 0:
            save_checkpoint(checkpoint_path, ddp_model.module, optimizer, epoch, world_size, comm_latency)

    total_time = time.time() - start_time_total
    total_time_tensor = torch.tensor([total_time], dtype=torch.float64, device=device)
    dist.all_reduce(total_time_tensor, op=dist.ReduceOp.MAX)
    total_time = total_time_tensor.item()

    if rank == 0:
        latency_path = None
        if comm_latency is not None:
            results_dir = os.path.join(work_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            latency_path = os.path.join(results_dir, f"latency_ws{world_size}.txt")
            with open(latency_path, "w") as f:
                f.write(f"{comm_latency:.6f}\n")

        checkpoints_dir = os.path.join(work_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        print("\nTraining selesai (DDP).")
        print(f"Total training time (DDP, world_size={world_size}, CPU): {total_time:.2f} detik")
        print(f"Model & checkpoint disimpan ke: {checkpoint_path}")
        print(f"Log performa disimpan ke: {results_path}")
        if latency_path:
            print(f"Latency (all_reduce) disimpan ke: {latency_path}")

        if monitor:
            monitor.stop()

        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
