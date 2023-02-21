#!/bin/bash
#SBATCH -J echo
#SBATCH -A pr28fa
#SBATCH --nodes=1
#SBATCH --partition=tmp0 --qos=nolimit

module load slurm_setup
source ../LRZoneAPI.sh # Takes care of modules

./echo

exit
