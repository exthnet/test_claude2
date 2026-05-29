#!/bin/bash
#SBATCH -p flash
#SBATCH -t 5:00
#SBATCH -o compare_%j.out
#SBATCH -e compare_%j.err

date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/25.9
nvidia-smi -L
echo "nproc = $(nproc)"

# overlap差が見えるよう CPU処理が無視できないサイズにする
N=512
NSET=4
NSET_GPU=2
NITER=5

echo "=== build (N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER) ==="
make clean
make acc  N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER || echo "BUILD FAIL: acc"
make acc2 N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER || echo "BUILD FAIL: acc2"

# 2版で同一のOpenMP環境にする
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo "=== run: current (matmul_acc) x2 ==="
./matmul_acc
./matmul_acc

echo "=== run: decoupled (matmul_acc2) x2 ==="
./matmul_acc2
./matmul_acc2

date
