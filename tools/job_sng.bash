#!/bin/bash
#SBATCH -J echo
#SBATCH -A <yourProject>
#SBATCH --nodes=1
#SBATCH --partition=test -t 30

module purge
module load slurm_setup
module load spack/23
module load intel-toolkit
source ../LRZoneAPI.sh # Takes care of modules

./dpecho_gpu

exit
