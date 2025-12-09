"""
Modul Monitoring Module untuk memenuhi komponen pemantauan pada tugas besar.
Mencatat CPU %, memori %, dan I/O jaringan ke CSV secara periodik.

Catatan: membutuhkan psutil
    pip install psutil
"""

import csv
import os
import tempfile
import threading
import time
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - fallback jika psutil belum terpasang
    psutil = None


def _default_base_dir() -> str:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = tempfile.gettempdir()
    work_dir = os.path.join(base, "komputasipararel", "UAS")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


class SystemMonitor:
    """
    Monitoring ringan berbasis psutil (CPU, RAM, net I/O) untuk logging ke CSV.
    Digunakan untuk memenuhi komponen "Monitoring Module" pada tugas besar.
    """

    def __init__(self, results_dir: Optional[str] = None) -> None:
        self.results_dir = results_dir or os.path.join(_default_base_dir(), "results", "monitoring")
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._csv_file = None
        self._csv_writer = None
        self._start_time: float = 0.0

    def start(self, run_name: str, interval_sec: float = 1.0) -> None:
        if psutil is None:
            print("[Monitoring] psutil tidak ditemukan. Jalankan: pip install psutil")
            return

        os.makedirs(self.results_dir, exist_ok=True)
        outfile = os.path.join(self.results_dir, f"monitor_{run_name}.csv")

        self._stop_event.clear()
        self._start_time = time.time()
        self._csv_file = open(outfile, mode="w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestamp", "relative_sec", "cpu_percent", "memory_percent", "net_bytes_sent", "net_bytes_recv"])
        self._csv_file.flush()

        # Prime CPU percent measurement
        psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(
            target=self._loop,
            args=(interval_sec,),
            daemon=True,
        )
        self._thread.start()
        print(f"[Monitoring] start -> {outfile}")

    def stop(self) -> None:
        if psutil is None:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
        self._thread = None
        self._csv_file = None
        self._csv_writer = None
        print("[Monitoring] stop")

    def _loop(self, interval_sec: float) -> None:
        while not self._stop_event.is_set():
            ts = time.time()
            rel = ts - self._start_time
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            net = psutil.net_io_counters()
            self._csv_writer.writerow([f"{ts:.3f}", f"{rel:.3f}", f"{cpu:.2f}", f"{mem:.2f}", net.bytes_sent, net.bytes_recv])
            self._csv_file.flush()
            time.sleep(interval_sec)
