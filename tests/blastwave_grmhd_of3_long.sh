#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
build_if_needed grmhd -DPHYSICS=GRMHD
run_test blastwave_grmhd_of3_long grmhd 1 blastwave_grmhd_of3_long.par
