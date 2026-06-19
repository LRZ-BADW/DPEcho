# Next Features (2026-06-19)

## 1. Restart scheme
Moderate. Save/load primitives + timestep state using existing MPI I/O.
Minimal viable: dump `v[8]` arrays + step counter + wall time, read back to restart.

## 2. VTK output
Small. Add `writeVTK()` alongside existing `writeBOV()`. Legacy `.vtk` structured
grid format is straightforward; `.vtu` (XML) is more involved.

## 3. Kerr metric
Very large. Kerr-Schild metric is coded in `Metric.hpp` but:
- Behind `#warning "not tested... expected to fail!"`
- Source terms in `MHD.cpp`/`GRMHD.cpp` return zero for non-Cartesian
- No BH problem ICs exist (Bondi, Fishbone-Moncrief torus, etc.)
- Spherical coordinates (r, θ, φ) need axis BCs and proper domain setup

## 4. UCT / Constrained Transport
Very large. CMake `UCT` option exists and allocates face arrays (`apG`, `amG`,
`vt1`, `vt2`) but they are never used. Full CT needs:
- Staggered face-centered B
- Edge-centered electric field from face fluxes
- Curl update for B (preserves div(B)=0 exactly)
- Face-to-cell interpolation for primitive recovery
- Reference: Gardiner & Stone 2005/2008, Balsara & Spicer 1999
