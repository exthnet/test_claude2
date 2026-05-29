#!/bin/bash
#SBATCH -p flash
#SBATCH -t 2:00
#SBATCH -o diag_%j.out
#SBATCH -e diag_%j.err

. /etc/profile.d/modules.sh 2>/dev/null
module load nvhpc/25.9 2>/dev/null

echo "=== node ==="; hostname
echo "=== gcc ==="; gcc --version 2>&1 | head -1; which gcc
echo "=== gcc 11 path ==="; ls /usr/lib/gcc/x86_64-linux-gnu/11/ 2>&1 | head
echo "=== gcc 13 path ==="; ls /usr/lib/gcc/x86_64-linux-gnu/13/ 2>&1 | head

printf 'int main(void){return 0;}\n' > /tmp/_t_$$.c

echo "=== nvc (default) ==="
nvc -O2 -o /tmp/_t_$$ /tmp/_t_$$.c 2>&1; echo "exit=$?"

echo "=== nvc -ccbin=gcc ==="
nvc -O2 -ccbin=$(which gcc) -o /tmp/_t2_$$ /tmp/_t_$$.c 2>&1; echo "exit=$?"

rm -f /tmp/_t_$$.c /tmp/_t_$$ /tmp/_t2_$$
