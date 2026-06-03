#!/bin/bash
# run_benchmark.sh
# Jalankan semua benchmark inference time di RunPod
# Usage: bash run_benchmark.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

WARMUP=10
RUNS=50

echo "======================================================"
echo "  BENCHMARK INFERENCE TIME - SEMUA MODEL SIBI"
echo "  Warmup: $WARMUP  |  Runs: $RUNS"
echo "======================================================"

# Install dependency jika belum ada
pip install -q opencv-python decord transformers timm

echo ""
echo "====== [1/8] VideoMAE ViT-B K400 (dummy) ======"
python benchmark_vitb_k400.py --warmup $WARMUP --runs $RUNS

echo ""
echo "====== [2/8] VideoMAE ViT-S SSv2 (dummy) ======"
python benchmark_vits_ssv2.py --warmup $WARMUP --runs $RUNS

echo ""
echo "====== [3/8] ViViT-B/16x2 (dummy) ======"
python benchmark_vivit.py --warmup $WARMUP --runs $RUNS

echo ""
echo "====== [4/8] I3D RGB (dummy) ======"
python benchmark_i3d.py --warmup $WARMUP --runs $RUNS

echo ""
echo "====== [5/8] VideoMAE ViT-B K400 (video asli) ======"
python benchmark_vitb_k400_realvideo.py --warmup $WARMUP --runs $RUNS

echo ""
echo "====== [6/8] VideoMAE ViT-S SSv2 (video asli) ======"
python benchmark_vits_ssv2_realvideo.py --warmup $WARMUP --runs $RUNS

echo ""
echo "====== [7/8] ViViT-B/16x2 (video asli) ======"
python benchmark_vivit_realvideo.py --warmup $WARMUP --runs $RUNS

echo ""
echo "====== [8/8] I3D RGB (video asli) ======"
python benchmark_i3d_realvideo.py --warmup $WARMUP --runs $RUNS

echo ""
echo "======================================================"
echo "  SELESAI"
echo "======================================================"
