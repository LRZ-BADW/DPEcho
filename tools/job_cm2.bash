#!/bin/bash
#SBATCH -J echo
#SBATCH --clusters=cm2_tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=25

module load slurm_setup

source ../LRZoneAPI.sh # Takes care of modules
module load vtune && source $VTUNE_PROFILER_2021_DIR/apsvars.sh
# mpiexec -n $SLURM_NTASKS ./echo

mpirun -n 1 aps ./echo
