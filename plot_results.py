import argparse
import glob
import os
import re
import tempfile
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REQUIRED_COLUMNS = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time_sec"]
DEFAULT_TRAIN_SAMPLES = 50000  # CIFAR-10 train set size


def get_writable_base_dir() -> str:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = tempfile.gettempdir()
    work_dir = os.path.join(base, "komputasipararel", "UAS")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hitung speedup/efficiency, throughput, dan plot hasil CIFAR-10 (single vs DDP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Folder yang berisi CSV log (default: <LOCALAPPDATA>/komputasipararel/UAS/results).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder untuk menyimpan grafik (default: <results-dir>/plots).",
    )
    parser.add_argument(
        "--single-csv",
        type=str,
        default=None,
        help="Path CSV baseline single node (default: single_node_lenet5_cifar10.csv di results-dir).",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path untuk menyimpan tabel ringkas (default: <results-dir>/summary_metrics.csv).",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help="Jumlah sampel training per epoch (dipakai untuk hitung throughput).",
    )
    parser.add_argument("--dpi", type=int, default=160, help="DPI untuk file PNG.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    results_dir = args.results_dir or os.path.join(get_writable_base_dir(), "results")
    output_dir = args.output_dir or os.path.join(results_dir, "plots")
    single_csv = args.single_csv or os.path.join(results_dir, "single_node_lenet5_cifar10.csv")
    summary_csv = args.summary_csv or os.path.join(results_dir, "summary_metrics.csv")

    if not os.path.isdir(results_dir):
        raise SystemExit(f"Folder results tidak ditemukan: {results_dir}")

    os.makedirs(output_dir, exist_ok=True)
    return results_dir, output_dir, single_csv, summary_csv


def find_ddp_logs(results_dir: str) -> List[Tuple[int, str]]:
    pattern = os.path.join(results_dir, "ddp_ws*_lenet5_cifar10.csv")
    paths: List[Tuple[int, str]] = []

    for path in glob.glob(pattern):
        name = os.path.basename(path)
        match = re.search(r"ddp_ws(\d+)_", name)
        if match:
            paths.append((int(match.group(1)), path))

    paths.sort(key=lambda item: item[0])
    return paths


def load_log(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Tidak ditemukan CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang di {csv_path}: {', '.join(missing)}")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"Nilai tidak valid/NaN di {csv_path}")

    return df


def load_metrics_from_csv(
    csv_path: str,
    world_size: int,
    train_samples: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = load_log(csv_path)
    epochs = len(df)
    total_time = float(df["epoch_time_sec"].sum())

    throughput_img_per_sec = 0.0
    if total_time > 0:
        # Approx global throughput: (jumlah sampel per epoch x jumlah epoch) dibagi total waktu.
        throughput_img_per_sec = (train_samples * epochs) / total_time

    avg_epoch_time_sec = total_time / epochs if epochs > 0 else 0.0  # Rata-rata durasi per epoch.
    final_test_acc = float(df["test_acc"].iloc[-1])
    final_test_loss = float(df["test_loss"].iloc[-1])

    metrics = {
        "world_size": world_size,
        "epochs": epochs,
        "total_time_sec": total_time,
        "throughput_img_per_sec": throughput_img_per_sec,
        "avg_epoch_time_sec": avg_epoch_time_sec,
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss,
    }
    return df, metrics


def plot_speedup(summary_df: pd.DataFrame, output_path: str, dpi: int) -> None:
    fig, ax = plt.subplots()
    ax.plot(summary_df["world_size"], summary_df["speedup"], marker="o", label="Speedup (DDP)")
    ax.plot(
        summary_df["world_size"],
        summary_df["world_size"],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Ideal linear",
    )
    ax.set_xlabel("World size (P)")
    ax.set_ylabel("Speedup (T_single / T_ddp)")
    ax.set_xticks(summary_df["world_size"])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def plot_efficiency(summary_df: pd.DataFrame, output_path: str, dpi: int) -> None:
    fig, ax = plt.subplots()
    ax.plot(summary_df["world_size"], summary_df["efficiency"] * 100.0, marker="o", label="Efficiency")
    ax.axhline(100.0, color="gray", linestyle="--", linewidth=1, label="Ideal 100%")
    ax.set_xlabel("World size (P)")
    ax.set_ylabel("Efficiency (%)")
    ax.set_xticks(summary_df["world_size"])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def plot_throughput(summary_df: pd.DataFrame, output_path: str, dpi: int) -> None:
    fig, ax = plt.subplots()
    ax.plot(summary_df["world_size"], summary_df["throughput_img_per_sec"], marker="o", label="Throughput")
    ax.set_xlabel("World size (P)")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_xticks(summary_df["world_size"])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def plot_accuracy(
    single_df: pd.DataFrame,
    ddp_runs: Sequence[Tuple[int, pd.DataFrame]],
    output_path: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        single_df["epoch"],
        single_df["test_acc"] * 100.0,
        label="Single (P=1)",
        linewidth=2,
    )

    for world_size, df in ddp_runs:
        ax.plot(
            df["epoch"],
            df["test_acc"] * 100.0,
            marker="o",
            linestyle="--",
            label=f"DDP P={world_size}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def plot_loss(
    single_df: pd.DataFrame,
    ddp_runs: Sequence[Tuple[int, pd.DataFrame]],
    output_path: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        single_df["epoch"],
        single_df["test_loss"],
        label="Single (P=1)",
        linewidth=2,
    )

    for world_size, df in ddp_runs:
        ax.plot(
            df["epoch"],
            df["test_loss"],
            marker="o",
            linestyle="--",
            label=f"DDP P={world_size}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def plot_monitoring_file(csv_path: str, output_path: str, dpi: int) -> bool:
    """
    Plot sederhana untuk Monitoring Module (CPU/RAM) guna melihat korelasi beban node.
    """
    if not os.path.isfile(csv_path):
        return False

    df = pd.read_csv(csv_path)
    required = ["relative_sec", "cpu_percent", "memory_percent"]
    if any(col not in df.columns for col in required):
        return False

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[required].isna().any().any():
        return False

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax1.plot(df["relative_sec"], df["cpu_percent"], label="CPU %")
    ax1.set_ylabel("CPU (%)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(df["relative_sec"], df["memory_percent"], color="orange", label="Mem %")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Memory (%)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    return True


def main() -> None:
    args = parse_args()
    results_dir, output_dir, single_csv, summary_csv = resolve_paths(args)
    train_samples = int(args.train_samples)

    single_df, single_metrics = load_metrics_from_csv(
        single_csv,
        world_size=1,
        train_samples=train_samples,
    )
    baseline_time = single_metrics["total_time_sec"]

    ddp_paths = find_ddp_logs(results_dir)
    if not ddp_paths:
        raise SystemExit(f"Tidak ada file ddp_ws*_lenet5_cifar10.csv di {results_dir}")

    ddp_runs: List[Tuple[int, pd.DataFrame]] = []
    summary_rows: List[Dict[str, float]] = [single_metrics]

    for world_size, path in ddp_paths:
        if world_size <= 1:
            continue
        df, metrics = load_metrics_from_csv(
            path,
            world_size=world_size,
            train_samples=train_samples,
        )
        ddp_runs.append((world_size, df))
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows).sort_values("world_size").reset_index(drop=True)
    summary_df["speedup"] = baseline_time / summary_df["total_time_sec"]
    summary_df["efficiency"] = summary_df["speedup"] / summary_df["world_size"]

    columns_out = [
        "world_size",
        "total_time_sec",
        "speedup",
        "efficiency",
        "throughput_img_per_sec",
        "avg_epoch_time_sec",
        "final_test_acc",
    ]
    summary_df.to_csv(summary_csv, index=False, columns=columns_out)

    print("Baseline (single node):")
    print(f"  CSV: {single_csv}")
    print(f"  total_time_sec = {baseline_time:.3f}")
    print(f"  final_test_acc = {single_metrics['final_test_acc']*100:.2f}%")

    print("\nDDP summary (relative to single):")
    display_cols = columns_out
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nTabel ringkas disimpan ke: {summary_csv}")

    speedup_png = os.path.join(output_dir, "speedup_vs_world_size.png")
    efficiency_png = os.path.join(output_dir, "efficiency_vs_world_size.png")
    throughput_png = os.path.join(output_dir, "throughput_vs_world_size.png")
    accuracy_png = os.path.join(output_dir, "test_accuracy_vs_epoch.png")
    loss_png = os.path.join(output_dir, "test_loss_vs_epoch.png")

    plot_speedup(summary_df, speedup_png, dpi=args.dpi)
    plot_efficiency(summary_df, efficiency_png, dpi=args.dpi)
    plot_throughput(summary_df, throughput_png, dpi=args.dpi)
    plot_accuracy(single_df, ddp_runs, accuracy_png, dpi=args.dpi)
    plot_loss(single_df, ddp_runs, loss_png, dpi=args.dpi)

    monitoring_plots: List[str] = []
    monitoring_dir = os.path.join(results_dir, "monitoring")
    if os.path.isdir(monitoring_dir):
        for csv_path in glob.glob(os.path.join(monitoring_dir, "monitor_*.csv")):
            name = os.path.splitext(os.path.basename(csv_path))[0]
            out_png = os.path.join(output_dir, f"{name}.png")
            if plot_monitoring_file(csv_path, out_png, dpi=args.dpi):
                monitoring_plots.append(out_png)

    print("File grafik:")
    print(f"  {speedup_png}")
    print(f"  {efficiency_png}")
    print(f"  {throughput_png}")
    print(f"  {accuracy_png}")
    print(f"  {loss_png}")
    if monitoring_plots:
        for p in monitoring_plots:
            print(f"  {p}")


if __name__ == "__main__":
    main()
