#!/usr/bin/env bash
set -eo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
BLD="$ROOT/build/grmhd"

bin=$(find "$BLD" -maxdepth 1 -name 'dpecho_*' -type f -executable 2>/dev/null | head -1)
if [[ -n "$bin" ]]; then
  cmake --build "$BLD" -j"$(nproc)" 2>&1 | tail -1
else
  cmake -S "$ROOT" -B "$BLD" -DCMAKE_CXX_COMPILER=mpiicpx -DCMAKE_BUILD_TYPE=Release -DSYCL_DEVICE=DEF -DMETRIC=CARTESIAN -DNRK=3 -DPHYSICS=GRMHD 2>&1 | tail -1
  cmake --build "$BLD" -j"$(nproc)" 2>&1 | tail -1
fi
bin=$(find "$BLD" -maxdepth 1 -name 'dpecho_*' -type f -executable | head -1)

cd "$ROOT"
mpirun -np 8 "$bin" tests/par/grmhd_mpi.par > "$DIR/out/grmhd_mpi.out" 2> "$DIR/out/grmhd_mpi.perf"

extract_dt()    { grep -oP 'dt \K[0-9.eE+\-]+' "$1" 2>/dev/null; }
extract_mcups() { awk '{print $NF}' "$1" 2>/dev/null; }

R_OUT="$DIR/ref/grmhd_mpi.out";  G_OUT="$DIR/out/grmhd_mpi.out"
R_PERF="$DIR/ref/grmhd_mpi.perf"; G_PERF="$DIR/out/grmhd_mpi.perf"
[[ ! -f "$R_OUT" ]]  && cp "$G_OUT" "$R_OUT"
[[ ! -f "$R_PERF" ]] && cp "$G_PERF" "$R_PERF"

mapfile -t dt_old < <(extract_dt "$R_OUT");   mapfile -t dt_new < <(extract_dt "$G_OUT")
mapfile -t mc_old < <(extract_mcups "$R_PERF"); mapfile -t mc_new < <(extract_mcups "$G_PERF")

nrows=${#dt_new[@]}; nmc=${#mc_old[@]}
echo "Step    dt_ref              dt_now              mcups_ref            mcups_now"
echo "────    ────                ────                ──────               ────────"
for ((i=0; i<nrows; i++)); do
  mi=$((i - 1))
  if (( mi >= 0 && mi < nmc )); then
    printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" \
      "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" "${mc_old[$mi]:-—}" "${mc_new[$mi]:-—}"
  else
    printf "%-6d  %-18s  %-18s  %-18s  %-18s\n" "$i" \
      "${dt_old[$i]:-—}" "${dt_new[$i]:-—}" "—" "—"
  fi
done
