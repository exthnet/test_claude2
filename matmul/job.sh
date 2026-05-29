#!/bin/bash
#SBATCH -p flash
#SBATCH -t 10:00
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

date
hostname

. /etc/profile.d/modules.sh
module load nvhpc/25.9

nvidia-smi

# 計算規模（必要に応じて変更）
N=1024
NSET=4
NSET_GPU=2
NITER=10          # GPU2版の反復回数
CPU_NITER=2       # CPU基準版は低速なため反復回数を縮小（GFLOPSはNITERで正規化され比較は公平）

echo "=== build ==="
make clean
make      N=$N NSET=$NSET NITER=$CPU_NITER                   # CPU基準版 (gcc)
make acc  N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER    # OpenACC + OpenMP (nvc)
make cuda N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER    # CUDA/cuBLAS + OpenMP (nvcc)

echo "=== run: CPU (matmul) ==="
./matmul

echo "=== run: OpenACC (matmul_acc) ==="
./matmul_acc

echo "=== run: CUDA/cuBLAS (matmul_cuda) ==="
./matmul_cuda

date
