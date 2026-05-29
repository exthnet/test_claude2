#!/bin/bash
#SBATCH -p flash
#SBATCH -t 5:00
#SBATCH -o compare3_%j.out
#SBATCH -e compare3_%j.err

date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/25.9
nvidia-smi -L
echo "nproc = $(nproc)"

N=512
NSET=4
NSET_GPU=2
NITER=5

echo "=== build (N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER) ==="
make clean
make acc  N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER || echo "BUILD FAIL: acc"
make acc2 N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER || echo "BUILD FAIL: acc2"
make acc3 N=$N NSET=$NSET NSET_GPU=$NSET_GPU NITER=$NITER || echo "BUILD FAIL: acc3"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo "=== current (matmul_acc) ==="
./matmul_acc
./matmul_acc
echo "=== decoupled-nested (matmul_acc2) ==="
./matmul_acc2
./matmul_acc2
echo "=== decoupled-flat (matmul_acc3) ==="
./matmul_acc3
./matmul_acc3

date
