#!/bin/bash
#SBATCH -p flash
#SBATCH -t 3:00
#SBATCH -o diag2_%j.out
#SBATCH -e diag2_%j.err

. /etc/profile.d/modules.sh 2>/dev/null
module load nvhpc/25.9 2>/dev/null

cd "$SLURM_SUBMIT_DIR" || exit 1
hostname

echo "=== (A) 実際の make acc を再現（全エラー取得） ==="
nvc -O2 -acc=gpu -mp=multicore -Minfo=accel,mp \
    -DN=256 -DNSET=4 -DNSET_GPU=2 -DNITER=1 \
    -o /tmp/_acc_$$ matmul_acc.c 2>&1
echo "exit=$?"
rm -f /tmp/_acc_$$

echo
echo "=== (B) localrc を再生成して NVLOCALRC で再試行 ==="
NVBIN=$(dirname "$(which nvc)")
LRC=/tmp/localrc_$$
makelocalrc "$NVBIN" -x -gcc "$(which gcc)" -gpp "$(which g++)" -g77 "$(which gfortran 2>/dev/null || echo /usr/bin/false)" -o "$LRC" 2>&1
echo "makelocalrc exit=$?"
echo "--- generated localrc (head) ---"; head -20 "$LRC" 2>&1

echo "--- rebuild with NVLOCALRC ---"
NVLOCALRC="$LRC" nvc -O2 -acc=gpu -mp=multicore -Minfo=accel,mp \
    -DN=256 -DNSET=4 -DNSET_GPU=2 -DNITER=1 \
    -o /tmp/_acc2_$$ matmul_acc.c 2>&1
echo "exit=$?"
rm -f /tmp/_acc2_$$ "$LRC"
