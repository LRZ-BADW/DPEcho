#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
build_if_needed grhd -DPHYSICS=GRHD
run_test uniform_grhd_per_long grhd 1 uniform_grhd_per_long.par
