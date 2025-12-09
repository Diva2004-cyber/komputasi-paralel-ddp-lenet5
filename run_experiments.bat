@echo off
setlocal

rem Jalankan semua eksperimen (single + DDP world size 2,3,4) dan plot hasil.
rem Opsional argumen:
rem   %1 = jumlah epoch (default: 10)

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" || exit /b 1

set "EPOCHS=%~1"
if "%EPOCHS%"=="" set "EPOCHS=10"

rem Pastikan python tersedia
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python tidak ditemukan di PATH.
    exit /b 1
)

rem Deteksi torchrun (fallback ke python -m torch.distributed.run)
where torchrun >nul 2>nul
if errorlevel 1 (
    set "TORCHRUN=python -m torch.distributed.run"
    echo [INFO] torchrun tidak ditemukan, pakai python -m torch.distributed.run
    python - <<EOF
import importlib.util, sys
spec = importlib.util.find_spec("torch.distributed.run")
sys.exit(0 if spec else 1)
EOF
    if errorlevel 1 (
        echo [ERROR] Modul torch.distributed.run tidak ditemukan. Pastikan PyTorch ter-install.
        exit /b 1
    )
    goto :start_jobs
) else (
    set "TORCHRUN=torchrun"
)

:start_jobs
echo ===========================================
echo [RUN] Single-process baseline
echo ===========================================
python train_single.py --epochs %EPOCHS%
if errorlevel 1 goto :error

for %%P in (2 3 4) do (
    echo ===========================================
    echo [RUN] DDP world size = %%P
    echo ===========================================
    %TORCHRUN% --standalone --nnodes=1 --nproc-per-node=%%P train_ddp.py --epochs %EPOCHS%
    if errorlevel 1 goto :error
)

echo ===========================================
echo [RUN] Plotting results
echo ===========================================
python plot_results.py
if errorlevel 1 goto :error

echo.
echo [DONE] Semua eksperimen selesai. Lihat folder results/ dan results/plots.
popd
exit /b 0

:error
echo.
echo [FAILED] Ada langkah yang gagal. Cek log di atas.
popd
exit /b 1
