"""
Prometheus exporter sederhana berbasis psutil untuk monitoring node (CPU, RAM, net I/O).
Dipakai sebagai bonus Monitoring Dashboard (bisa di-scrape Prometheus lalu divisualisasikan di Grafana).

Persiapan:
    pip install psutil prometheus-client

Jalankan:
    python monitor_exporter.py --port 8000 --interval 1.0 --run-name ddp_ws4

Lalu tambahkan job scrape di Prometheus (contoh):
  - job_name: "uas_ddp"
    static_configs:
      - targets: ["127.0.0.1:8000"]
        labels:
          run: "ddp_ws4"
"""

import argparse
import sys
import time

try:
    import psutil
except ImportError:
    psutil = None

try:
    from prometheus_client import Gauge, start_http_server
except ImportError:
    Gauge = None
    start_http_server = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prometheus exporter untuk monitoring CPU/RAM/net I/O (psutil)."
    )
    parser.add_argument("--port", type=int, default=8000, help="Port HTTP exporter")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval update (detik)")
    parser.add_argument("--run-name", type=str, default="default", help="Label run/eksperimen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if psutil is None or Gauge is None or start_http_server is None:
        print("[Monitoring] psutil atau prometheus-client belum terpasang.")
        print("Instal terlebih dahulu: pip install psutil prometheus-client")
        sys.exit(1)

    # Inisialisasi gauges dengan label run
    cpu_gauge = Gauge("node_cpu_percent", "CPU usage percent", ["run"])
    mem_gauge = Gauge("node_memory_percent", "Memory usage percent", ["run"])
    net_sent_gauge = Gauge("node_net_bytes_sent", "Network bytes sent (cumulative)", ["run"])
    net_recv_gauge = Gauge("node_net_bytes_recv", "Network bytes recv (cumulative)", ["run"])

    start_http_server(args.port)
    print(f"[Monitoring] Prometheus exporter berjalan di port {args.port} (run={args.run_name})")

    # Prime CPU percent
    psutil.cpu_percent(interval=None)

    while True:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        net = psutil.net_io_counters()

        cpu_gauge.labels(run=args.run_name).set(cpu)
        mem_gauge.labels(run=args.run_name).set(mem)
        net_sent_gauge.labels(run=args.run_name).set(net.bytes_sent)
        net_recv_gauge.labels(run=args.run_name).set(net.bytes_recv)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
