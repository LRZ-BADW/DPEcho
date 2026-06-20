# Next Features (2026-06-20)

## Done
- Restart from VTK dumps: reads `.vti`, restores primitives + metadata, calls `prim2cons`
- VTK XML output (`.vti` + `.pvti`) fully replaces BOV
- BOV has been removed entirely; legacy utility `tools/bov-split.py` deleted

## 1. Kerr metric
Very large. Kerr-Schild metric is coded in `Metric.hpp` but:
- Behind `#warning "not tested... expected to fail!"`
- Source terms in `MHD.cpp`/`GRMHD.cpp` return zero for non-Cartesian
- No BH problem ICs exist (Bondi, Fishbone-Moncrief torus, etc.)
- Spherical coordinates (r, θ, φ) need axis BCs and proper domain setup

## 2. UCT / Constrained Transport
Very large. CMake `UCT` option exists and allocates face arrays (`apG`, `amG`,
`vt1`, `vt2`) but they are never used. Full CT needs:
- Staggered face-centered B
- Edge-centered electric field from face fluxes
- Curl update for B (preserves div(B)=0 exactly)
- Face-to-cell interpolation for primitive recovery
- Reference: Gardiner & Stone 2005/2008, Balsara & Spicer 1999
