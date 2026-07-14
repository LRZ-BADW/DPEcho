#!/usr/bin/env bash
set -eo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
mkdir -p "$DIR/out"

build_if_needed() {
  local name="$1"; shift
  local bld="$ROOT/build/$name"
  local bin=$(find "$bld" -maxdepth 1 -name 'dpecho_*' -type f -executable 2>/dev/null | head -1)
  if [[ -n "$bin" ]]; then
    cmake --build "$bld" -j"$(nproc)" 2>&1 | tail -1
  else
    cmake -S "$ROOT" -B "$bld" -DCMAKE_CXX_COMPILER=mpiicpx -DCMAKE_BUILD_TYPE=Release -DSYCL_DEVICE=DEF -DMETRIC=CARTESIAN -DNRK=3 "$@" 2>&1 | tail -1
    cmake --build "$bld" -j"$(nproc)" 2>&1 | tail -1
  fi
}
bin_name() { find "$ROOT/build/$1" -maxdepth 1 -name 'dpecho_*' -type f -executable | head -1; }
extract_dt()   { grep -oP 'dt \K[0-9.eE+\-]+' "$1" 2>/dev/null; }
extract_mcups() { awk '{print $NF}' "$1" 2>/dev/null; }

run_test() {
  local test="$1" bld="$2" ranks="$3" par="$4"
  local bin=$(bin_name "$bld"); cd "$ROOT"
  mpirun -np "$ranks" "$bin" "$DIR/par/$par" > "$DIR/out/$test.out" 2> "$DIR/out/$test.perf"
  local R_OUT="$DIR/ref/$test.out" R_PERF="$DIR/ref/$test.perf"
  local G_OUT="$DIR/out/$test.out" G_PERF="$DIR/out/$test.perf"
  [[ ! -f "$R_OUT" ]] && cp "$G_OUT" "$R_OUT" && echo "[$test] Recorded ref stdout"
  [[ ! -f "$R_PERF" ]] && cp "$G_PERF" "$R_PERF" && echo "[$test] Recorded ref stderr"
  mapfile -t dt_old < <(extract_dt "$R_OUT"); mapfile -t dt_new < <(extract_dt "$G_OUT")
  mapfile -t mc_old < <(extract_mcups "$R_PERF"); mapfile -t mc_new < <(extract_mcups "$G_PERF")
  local nrows=${#dt_new[@]} nmc=${#mc_old[@]}
  echo "Step    dt_ref              dt_now              mcups_ref            mcups_now"
  echo "────    ────                ────                ──────               ────────"
  for ((i=1; i<nrows; i++)); do
    mi=$((i - 1))
    if (( mi >= 0 && mi < nmc )); then
      printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" "${mc_old[$mi]:-—}" "${mc_new[$mi]:-—}"
    else
      printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" "—" "—"
    fi
  done
  echo
}
