#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

build_if_needed grmhd -DPHYSICS=GRMHD

for t in \
  "alfven_grmhd_per_long    grmhd 1  alfven_grmhd_per_long.par" \
  "alfven_grmhd_of3_long    grmhd 1  alfven_grmhd_of3_long.par" \
  "blastwave_grmhd_of0_long grmhd 1  blastwave_grmhd_of0_long.par" \
  "blastwave_grmhd_of3_long grmhd 1  blastwave_grmhd_of3_long.par" \
  "alfven_grmhd_of0         grmhd 8  alfven_grmhd_of0.par" \
  "alfven_grmhd_per         grmhd 1  alfven_grmhd_per.par"; do
  read -r test bld ranks par <<< "$t"
  echo "━━━ $test ━━━"
  run_test "$test" "$bld" "$ranks" "$par"
done
