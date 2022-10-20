#!/bin/bash

module unload devEnv intel-mpi intel-mkl intel
module load gcc/10

module swap intel-oneapi intel-oneapi/2021.2
module load mpi compiler
module swap intel-oneapi-vtune vtune

module load cmake/3.16.5
