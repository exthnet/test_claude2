#!/bin/bash
#SBATCH -p flash
#SBATCH -t 5:00
#SBATCH -o verify_%j.out
#SBATCH -e verify_%j.err

date
hostname

. /etc/profile.d/modules.sh
module load nvhpc/25.9

nvidia-smi -L

# 動作確認用の小さい問題サイズ（全版で同一 → checksum も一致するはず）
N=256
NSET=4
NSET_GPU=2
NITER=3

echo "=== build (N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER) ==="
make clean
make      N=$N NSET=$NSET NITER=$NITER                       || echo "BUILD FAIL: cpu"
make acc  N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER    || echo "BUILD FAIL: acc"
make cuda N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER    || echo "BUILD FAIL: cuda"

echo "=== run: CPU (matmul) ==="
./matmul       || echo "RUN FAIL: cpu"
echo "=== run: OpenACC (matmul_acc) ==="
./matmul_acc   || echo "RUN FAIL: acc"
echo "=== run: CUDA/cuBLAS (matmul_cuda) ==="
./matmul_cuda  || echo "RUN FAIL: cuda"

date
