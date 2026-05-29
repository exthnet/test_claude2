#!/bin/bash
#SBATCH -p flash
#SBATCH -t 10:00
date
hostname
. /etc/profile.d/modules.sh
module load nvhpc/25.9
nvidia-smi
./a.out
date
