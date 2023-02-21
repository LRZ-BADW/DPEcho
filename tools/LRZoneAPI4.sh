#!/bin/bash

module unload devEnv intel-mpi intel-mkl intel
module load gcc/10

module swap intel-oneapi intel-oneapi/2022.3
module swap vtune intel-oneapi-vtune

module load cmake
