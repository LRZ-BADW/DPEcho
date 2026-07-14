#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
build_if_needed mhd -DPHYSICS=MHD
run_test alfven_mhd_per_long mhd 1 alfven_mhd_per_long.par
