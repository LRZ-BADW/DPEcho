# DPEcho Intel Optimization Report Analysis

Generated from `-DCMAKE_BUILD_TYPE=OptReport` build (Intel oneAPI 2026.0.0, GRMHD physics).

---

## Top Priority

- [x] ### 1. `Metric` methods not inlined on host

`gCon()`, `dgCov()`, `dgBeta()` are called **30+ times each** in the unrolled
`physicalSource` loop but the compiler marks them EXTERN (not inlined). This is
the single biggest host-side optimization opportunity.

**Fix:** Add `inline` or `[[gnu::always_inline]]` to these methods in
`Metric.hpp`.

- [x] ### 1b. `Metric` convenience functions not inlined on host

`g3DCov()`, `g3DCon()`, `beta()`, `con2Cov()`, `cov2Con()` were EXTERN calls
defined only in `Metric.cpp`. For `physicalFlux` this meant computing gCov[9],
gCon[9], betai[3] through opaque function calls.

**Fix:** Moved all five from `Metric.cpp` to `Metric.hpp` as `inline`.

**Result:** `physicalFlux` spills reduced 40r/32s -> 30r/24s (-25%).
`physicalSource` spills reduced 39r/34s -> 21r/17s (-50%).

- [ ] ### 2. `cons2prim` Newton-Raphson cannot vectorize

- `GRMHD.cpp:54` — 5 FLOW dependencies across iterations
- `Problem.cpp:309`, `Problem.cpp:392` — 4 FLOW + 1 ANTI dependence

The iterative solve has true loop-carried dependencies that block SIMD.
Inherent to the Gauss-Seidel-style algorithm.

**Not a real bottleneck:** `cons2prim` is already called inside SYCL
`parallel_for` (Problem.cpp:337), so parallelism is at the cell level, not the
iteration level. The inner Newton-Raphson loop is per-cell and sequential by
nature. No action needed.

- [x] ### 3. `mp5()` register spilling (Solver.cpp)

69 reloads, 47 spills, 376 bytes spilled to stack. 55+ inlined calls
(mm2, mm4, sign, fabs, fmin, fmax) create extreme register pressure.

**Fix:** Split `mp5()` into fast-path (inline, common case) and `mp5_slow()`
(`__attribute__((noinline))`, rare hard case). Removed dead `res` variable.

**Result:** `mp5_slow` at 59r/40s is now isolated from the hot path.

---

## Medium Priority

### 4. `physicalFlux` / `physicalSource` register spilling

After Metric inlining, remaining spilling is:
- `physicalFlux`: 30 reloads, 24 spills, 632B stack, 6 available regs
- `physicalSource`: 21 reloads, 17 spills, 432B stack, 14 available regs

Root cause: large stack arrays (`gCov[9]`, `gCon[9]`, `betai[3]`, plus
`vCov`, `bCov`, `eCov`, `eCon`, `sCov`, `sCon`). Both physicalFlux calls
in the inner loop (`echo.cpp:142-143`) use the same `Metric g`, computing
identical metric arrays twice.

**Potential fix (deferred):** Hoist metric arrays out of `physicalFlux` and
pass pre-computed `gCov`, `gCon`, `betai` as parameters. Requires signature
change across GRMHD/GRHD/HD/MHD backends.

### 5. `physicalSource` vector dependencies (`GRMHD.cpp:175/178`)

Christoffel symbol computation creates ANTI + OUTPUT + FLOW dependencies in the
nested 3D index loops. Unrolled by 3 but not vectorized.

### 6. `Domain.cpp:127` output dependence

Communication indexing loop has write-write conflicts in halo exchange setup.
May benefit from temporary arrays.

---

## What Is Already Optimized

- `holibDer` / `holibRec` — successfully unrolled by 5-6x
- `physicalFlux` — unrolled by 8x, good SLP vectorization (64 groups)
- All `Metric` methods inlined (scalar, 3x3 fill, matVec)
- `mp5` split into fast-path + `mp5_slow(noinline)`
- `kTimer` / `Log::cups()` removed from `echo.cpp`
- Test scripts skip step 0 (loop starts at i=1)
- `OptReport` CMake build type for generating `.optrpt` files
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
