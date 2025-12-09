# Analisis Efisiensi Pelatihan Terdistribusi Model LeNet-5 pada Dataset CIFAR-10 Menggunakan PyTorch Distributed Data Parallel (DDP) Berbasis CPU
## Deskripsi Singkat / Overview
Repositori ini berisi sistem pelatihan CNN (LeNet-5) pada CIFAR-10 menggunakan PyTorch, dengan perbandingan antara pelatihan single node (single-process, CPU) dan distributed training memakai Distributed Data Parallel (DDP) berbasis CPU (backend `gloo`). Hasil pelatihan dicatat ke CSV dan divisualisasikan untuk menilai speedup, efficiency, throughput, serta kurva akurasi/loss.
## Konteks Akademik
- Mata kuliah: Komputasi Paralel dan Terdistribusi
- Tema tugas besar: Parallel Deep Learning Training System (Tema 1)
- Judul proyek: **"Analisis Efisiensi Pelatihan Terdistribusi Model LeNet-5 pada Dataset CIFAR-10 Menggunakan PyTorch Distributed Data Parallel (DDP) Berbasis CPU"**
- Tujuan: mengevaluasi efisiensi pelatihan terdistribusi di CPU (single node, multi-proses) dibanding baseline single-process.
## Fitur Utama
- Arsitektur LeNet-5 untuk CIFAR-10 (`model_lenet5.py`).
- Training single-process (baseline CPU) (`train_single.py`).
- Training DDP berbasis CPU, backend `gloo`, world size 1-4 (`train_ddp.py`).
- Logging CSV per epoch (loss/acc train-test, waktu per epoch, total waktu).
- Agregasi metrik + plotting speedup/efficiency/throughput dan kurva akurasi/loss (`plot_results.py` -> `results/plots/`).
- Skrip batch opsional untuk menjalankan seluruh eksperimen sekaligus (`run_experiments.bat`).
## Arsitektur & Teknologi
- Single physical node, multi-proses (data parallel) dengan world size 1-4.
- Backend komunikasi: `gloo` (CPU).
- Dataset CIFAR-10 tersimpan di folder `data/` (diarahkan ke lokasi tulis aman, mis. `<LOCALAPPDATA>/komputasipararel/UAS/data` pada Windows).
- Teknologi: Python 3.10+, PyTorch (CPU), torchvision, matplotlib, pandas.
## Struktur Direktori
- `data/` - CIFAR-10 (otomatis diunduh torchvision).
- `checkpoints/` - model tersimpan: `lenet5_single_cpu.pt`, `lenet5_ddp_ws{1..4}_cpu.pt`.
- `results/` - log dan ringkasan:
  - CSV: `single_node_lenet5_cifar10.csv`, `ddp_ws{1,2,3,4}_lenet5_cifar10.csv`, `summary_metrics.csv`.
  - `results/plots/`: `speedup_vs_world_size.png`, `efficiency_vs_world_size.png`, `throughput_vs_world_size.png`, `test_accuracy_vs_epoch.png`, `test_loss_vs_epoch.png`, dll.
- Script utama: `model_lenet5.py`, `train_single.py`, `train_ddp.py`, `plot_results.py`.
- Skrip bantuan: `run_experiments.bat`.
- `README.md` - dokumen ini.
## Kebutuhan Sistem & Dependensi
- Python 3.10+ (disarankan via Anaconda/Miniconda).
- PyTorch CPU + torchvision.
- matplotlib, pandas (scikit-learn opsional).
## Instalasi & Setup Environment (contoh conda)
```bash
conda create -n ddp-lenet python=3.10 -y
conda activate ddp-lenet
# dari root repo
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 cpuonly -c pytorch -y
pip install matplotlib pandas
```
## Cara Menjalankan Proyek
1) Baseline single-process (CPU):
```bash
python train_single.py --epochs 10
```
2) DDP (CPU, backend `gloo`) - contoh world size 4:
```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 train_ddp.py --epochs 10
```
Ganti `--nproc-per-node` ke 2 atau 3 untuk world size lain.
3) Skrip batch (opsional, Windows):
```bash
run_experiments.bat 10
```
Menjalankan baseline, DDP (P=2,3,4), lalu plotting.
4) Plotting & ringkasan metrik:
```bash
python plot_results.py
# opsi: --train-samples 50000 --results-dir <path> --output-dir <path>
```
Output: `results/summary_metrics.csv` dan PNG di `results/plots/`.
## Desain Eksperimen & Metrik
- Konfigurasi diuji: world size P = 1, 2, 3, 4 (single node, multi-proses).
- Metrik:
  - Total training time per konfigurasi.
  - Speedup `S(P) = T_single / T_ddp(P)`.
  - Efficiency `E(P) = S(P) / P`.
  - Throughput `images/sec = (jumlah sampel train x epochs) / total_time`.
  - Avg epoch time `= total_time / epochs`.
  - Test accuracy/loss per epoch dan nilai akhir.
## Ringkasan Hasil Utama (pola)
- Speedup meningkat hingga P=4 tetapi sub-linear karena overhead DDP di CPU.
- Efficiency turun dari ~1.0 (P=1) ke sekitar ~0.5 (P=4).
- Throughput naik seiring world size, namun dengan diminishing returns.
- Akurasi test sedikit menurun saat world size bertambah (trade-off kecepatan vs akurasi).
## Cara Reproduksi Eksperimen
1. Clone repo dan masuk ke direktori proyek.
2. Buat/aktifkan environment, install dependensi (lihat setup).
3. Jalankan baseline:
   ```bash
   python train_single.py --epochs 10
   ```
4. Jalankan DDP untuk P=2,3,4:
   ```bash
   python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 train_ddp.py --epochs 10
   # ulangi untuk 3 dan 4
   ```
   atau gunakan `run_experiments.bat 10` (Windows).
5. Plot dan ringkas hasil:
   ```bash
   python plot_results.py
   ```
6. Lihat artefak:
   - CSV ringkasan: `results/summary_metrics.csv`
   - Grafik: `results/plots/*.png`
   - Model: `checkpoints/lenet5_*.pt`
   ## Monitoring Dashboard (Bonus Prometheus/Grafana)
- Ekspor metrik node (CPU, RAM, net I/O) dengan Prometheus exporter berbasis psutil:
  ```bash
  pip install psutil prometheus-client
  python monitor_exporter.py --port 8000 --interval 1.0 --run-name ddp_ws4
  ```
- Tambahkan job scrape Prometheus:
  ```
  - job_name: "uas_ddp"
    static_configs:
      - targets: ["127.0.0.1:8000"]
        labels:
          run: "ddp_ws4"
  ```
- Buat dashboard di Grafana dengan metrik:
  - `node_cpu_percent{run="ddp_ws4"}`
  - `node_memory_percent{run="ddp_ws4"}`
  - `node_net_bytes_sent{run="ddp_ws4"}` / `node_net_bytes_recv{run="ddp_ws4"}`