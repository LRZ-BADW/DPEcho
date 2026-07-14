#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
build_if_needed hd -DPHYSICS=HD
run_test uniform_hd_per_long hd 1 uniform_hd_per_long.par
