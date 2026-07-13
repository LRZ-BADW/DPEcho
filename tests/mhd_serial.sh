#!/usr/bin/env bash
set -eo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
BLD="$ROOT/build/mhd"

bin=$(find "$BLD" -maxdepth 1 -name 'dpecho_*' -type f -executable 2>/dev/null | head -1)
if [[ -n "$bin" ]]; then
  cmake --build "$BLD" -j"$(nproc)" 2>&1 | tail -1
else
  cmake -S "$ROOT" -B "$BLD" -DCMAKE_CXX_COMPILER=mpiicpx -DCMAKE_BUILD_TYPE=Release -DSYCL_DEVICE=DEF -DMETRIC=CARTESIAN -DNRK=3 -DPHYSICS=MHD 2>&1 | tail -1
  cmake --build "$BLD" -j"$(nproc)" 2>&1 | tail -1
fi
bin=$(find "$BLD" -maxdepth 1 -name 'dpecho_*' -type f -executable | head -1)

cd "$ROOT"
mpirun -np 1 "$bin" tests/par/mhd_periodic.par > "$DIR/out/mhd_serial.out" 2> "$DIR/out/mhd_serial.perf"

extract_dt()    { grep -oP 'dt \K[0-9.eE+\-]+' "$1" 2>/dev/null; }
extract_mcups() { awk '{print $NF}' "$1" 2>/dev/null; }

R_OUT="$DIR/ref/mhd_serial.out";  G_OUT="$DIR/out/mhd_serial.out"
R_PERF="$DIR/ref/mhd_serial.perf"; G_PERF="$DIR/out/mhd_serial.perf"
[[ ! -f "$R_OUT" ]]  && cp "$G_OUT" "$R_OUT"
[[ ! -f "$R_PERF" ]] && cp "$G_PERF" "$R_PERF"

mapfile -t dt_old < <(extract_dt "$R_OUT");   mapfile -t dt_new < <(extract_dt "$G_OUT")
mapfile -t mc_old < <(extract_mcups "$R_PERF"); mapfile -t mc_new < <(extract_mcups "$G_PERF")

nrows=${#dt_new[@]}; nmc=${#mc_old[@]}
echo "Step    dt_ref              dt_now              mcups_ref            mcups_now"
echo "────    ────                ────                ──────               ────────"
for ((i=1; i<nrows; i++)); do
  mi=$((i - 1))
  if (( mi >= 0 && mi < nmc )); then
    printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" \
      "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" "${mc_old[$mi]:-—}" "${mc_new[$mi]:-—}"
  else
    printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" \
      "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" "—" "—"
  fi
done
