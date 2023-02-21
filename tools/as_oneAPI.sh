
#!/bin/bash
module load oneapi
module load graphics-compute-runtime
module load mpi/2021.4

export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"

# If serial
export ZE_AFFINITY_MASK=0.0
# If MPI
export I_MPI_OFFLOAD_TOPOLIB=level_zero
export I_MPI_OFFLOAD_CELL=tile
export I_MPI_OFFLOAD_DOMAIN_SIZE=-1
export I_MPI_OFFLOAD_DEVICES=all

export I_MPI_DEBUG=3
