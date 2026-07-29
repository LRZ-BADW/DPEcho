#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
build_if_needed grmhd -DPHYSICS=GRMHD
run_test alfven_grmhd_of0 grmhd 8 alfven_grmhd_of0.par
