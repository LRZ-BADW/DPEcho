#!/usr/bin/env bash
set -eo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
mkdir -p "$DIR/out"

# ── Build helpers ──────────────────────────────────────────────────────────────
build_if_needed() {
  local name="$1"; shift
  local bld="$ROOT/build/$name"
  local bin=$(find "$bld" -maxdepth 1 -name 'dpecho_*' -type f -executable 2>/dev/null | head -1)
  if [[ -n "$bin" ]]; then
    cmake --build "$bld" -j"$(nproc)" 2>&1 | tail -1
  else
    cmake -S "$ROOT" -B "$bld" \
      -DCMAKE_CXX_COMPILER=mpiicpx \
      -DCMAKE_BUILD_TYPE=Release \
      -DSYCL_DEVICE=DEF \
      -DMETRIC=CARTESIAN -DNRK=3 \
      "$@" 2>&1 | tail -1
    cmake --build "$bld" -j"$(nproc)" 2>&1 | tail -1
  fi
}

bin_name() { find "$ROOT/build/$1" -maxdepth 1 -name 'dpecho_*' -type f -executable | head -1; }

# ── Build 4 binaries ──────────────────────────────────────────────────────────
build_if_needed grmhd -DPHYSICS=GRMHD
build_if_needed mhd   -DPHYSICS=MHD
build_if_needed hd    -DPHYSICS=HD
build_if_needed grhd  -DPHYSICS=GRHD

# ── Extraction helpers ────────────────────────────────────────────────────────
# stdout: "[main] Step # N: t T i.e. X% dt DT walltime/s W"
# stderr: "NRANKS GRIDSIZE WALLTIME SPEC TOTAL_SPEC"
extract_dt()   { grep -oP 'dt \K[0-9.eE+\-]+' "$1" 2>/dev/null; }
extract_mcups() { awk '{print $NF}' "$1" 2>/dev/null; }

# ── Run + compare ─────────────────────────────────────────────────────────────
run_and_compare() {
  local test="$1" bld="$2" ranks="$3" par="$4"
  local bin=$(bin_name "$bld")
  cd "$ROOT"
  mpirun -np "$ranks" "$bin" "$DIR/par/$par" \
    > "$DIR/out/$test.out" 2> "$DIR/out/$test.perf"

  local R_OUT="$DIR/ref/$test.out"
  local R_PERF="$DIR/ref/$test.perf"
  local G_OUT="$DIR/out/$test.out"
  local G_PERF="$DIR/out/$test.perf"

  # Record ref if missing
  if [[ ! -f "$R_OUT" ]]; then cp "$G_OUT" "$R_OUT"; echo "[$test] Recorded ref stdout"; fi
  if [[ ! -f "$R_PERF" ]]; then cp "$G_PERF" "$R_PERF"; echo "[$test] Recorded ref stderr"; fi

  # Extract
  mapfile -t dt_old  < <(extract_dt "$R_OUT")
  mapfile -t dt_new  < <(extract_dt "$G_OUT")
  mapfile -t mc_old  < <(extract_mcups "$R_PERF")
  mapfile -t mc_new  < <(extract_mcups "$G_PERF")

  local nrows=${#dt_new[@]}
  # mcups in stderr skip step 0 (if(iStep_) guard), so mcups[0] = step 1
  local nmc=${#mc_old[@]}

  echo "Step    dt_ref              dt_now              mcups_ref            mcups_now"
  echo "────    ────                ────                ──────               ────────"
  for ((i=1; i<nrows; i++)); do
    mi=$((i - 1))
    if (( mi >= 0 && mi < nmc )); then
      printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" \
        "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" \
        "${mc_old[$mi]:-—}" "${mc_new[$mi]:-—}"
    else
      printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" \
        "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" "—" "—"
    fi
  done
  echo
}

# ── Run 9 tests ───────────────────────────────────────────────────────────────
for t in \
  "alfven_grmhd_per_long    grmhd 1  alfven_grmhd_per_long.par" \
  "alfven_grmhd_of3_long    grmhd 1  alfven_grmhd_of3_long.par" \
  "alfven_mhd_per_long      mhd   1  alfven_mhd_per_long.par" \
  "uniform_hd_per_long      hd    1  uniform_hd_per_long.par" \
  "uniform_grhd_per_long    grhd  1  uniform_grhd_per_long.par" \
  "blastwave_grmhd_of0_long grmhd 1  blastwave_grmhd_of0_long.par" \
  "blastwave_grmhd_of3_long grmhd 1  blastwave_grmhd_of3_long.par" \
  "alfven_grmhd_of0         grmhd 8  alfven_grmhd_of0.par" \
  "alfven_grmhd_per         grmhd 1  alfven_grmhd_per.par"; do
  read -r test bld ranks par <<< "$t"
  echo "━━━ $test ━━━"
  run_and_compare "$test" "$bld" "$ranks" "$par"
done
