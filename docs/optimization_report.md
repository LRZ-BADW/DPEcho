# DPEcho Intel Optimization Report Analysis

Generated from `-DCMAKE_BUILD_TYPE=OptReport` build (Intel oneAPI 2026.0.0, GRMHD physics).

---

## Top Priority

### 1. `Metric` methods not inlined on host

`gCon()`, `dgCov()`, `dgBeta()` are called **30+ times each** in the unrolled
`physicalSource` loop but the compiler marks them EXTERN (not inlined). This is
the single biggest host-side optimization opportunity.

**Fix:** Add `inline` or `[[gnu::always_inline]]` to these methods in
`Metric.hpp`.

### 2. `cons2prim` Newton-Raphson cannot vectorize

- `GRMHD.cpp:54` — 5 FLOW dependencies across iterations
- `Problem.cpp:309`, `Problem.cpp:392` — 4 FLOW + 1 ANTI dependence

The iterative solve has true loop-carried dependencies that block SIMD.
Inherent to the Gauss-Seidel-style algorithm.

**Possible fix:** Rewrite with Jacobi-style (block) iteration, or use explicit
SIMD intrinsics.

### 3. `mp5()` register spilling (Solver.cpp)

69 reloads, 47 spills, 376 bytes spilled to stack. 55+ inlined calls
(mm2, mm4, sign, fabs, fmin, fmax) create extreme register pressure.

**Possible fix:** Split `mp5()` into smaller functions, or reduce intermediate
temporaries.

---

## Medium Priority

### 4. `physicalSource` vector dependencies (`GRMHD.cpp:175/178`)

Christoffel symbol computation creates ANTI + OUTPUT + FLOW dependencies in the
nested 3D index loops. Unrolled by 3 but not vectorized.

### 5. `Domain.cpp:127` output dependence

Communication indexing loop has write-write conflicts in halo exchange setup.
May benefit from temporary arrays.

---

## What Is Already Optimized

- `holibDer` / `holibRec` — successfully unrolled by 5-6x
- `physicalFlux` — unrolled by 8x, good SLP vectorization (66 groups)
- All SYCL device-side `Metric` methods properly inlined
- Redundant `memcpy` / `memset` eliminated by the optimizer
- `Domain.cpp:65` coordinate transformation — vectorized (speedup 1.11x)

---

## Build Instructions

```bash
cmake -S . -B build/grmhd_optreport \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_BUILD_TYPE=OptReport \
  -DMETRIC=CARTESIAN -DNRK=3 -DPHYSICS=GRMHD

cmake --build build/grmhd_optreport -j$(nproc)
```

Report files appear in `build/grmhd_optreport/CMakeFiles/dpecho.dir/src/*.optrpt`.
Host-side reports are `*.optrpt`, device-side reports are `*-spir64-*.optrpt`.
