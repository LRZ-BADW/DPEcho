//   Copyright(C) 2022 Alexander Pöppl, Intel Corporation
//   Copyright(C) 2022 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Metric_hpp_
#define _Metric_hpp_

#include "Logger.hpp"
#include "utils/tb-types.hpp"

#include "echo.hpp"

#include <sycl/sycl.hpp>

class Metric {
  private:
    [[maybe_unused]] real x[NDIM]; // Spatial coordinates

  public:
    Metric (real x0, real x1, real x2) : x{x0, x1, x2 } {} // Initialize x directly upon construction. Theory: not doing so causes issues with computations below.
#if METRIC > CARTESIAN
    static real bhm, bha, bhc;
    // read them from file on host (in main) and set here once...
    static inline void setParameters(real mm, real aa, real cc) {
      Metric::bhm = mm; Metric::bha = aa; Metric::bhc = cc;
#if METRIC == KERR_SCHILD
#warning "The Kerr-Schild Metric is not tested (may be incorrect!) and has no proper Problem case. It is expected to fail!"
      Log::cout(4) << TAG << "WARNING: The Kerr-Schild Metric is not tested (may be incorrect!) and has no proper Problem case. It is expected to fail!" << Log::endl;
#endif
    }
#else // For uniformity
    static inline void setParameters(real bhm, real bha, real bhc) {}
#endif

#if   METRIC == CARTESIAN   // Nothing needed
#elif METRIC == KERR_SCHILD // declare needed vars
  // WOW!!!
  const real r2 = x[0]*x[0], sint = sycl::sin(x[1]), sint2 = sint*sint, cost = sycl::cos(x[1]);
  const real a = bha, a2 = a*a, delta= r2-2.*bhm*x[0]+a2, rho2 = r2+a2*(1.-sint2), zz = 2.*bhm*x[0]/rho2;
  const real sigma = (r2+a2)*(r2+a2)-a2*delta*sint2, det = (1.+zz)*(sigma/rho2-a2*(1.+zz)*sint2);
  // For derivatives TODO: maybe make these into a function Metric::initDeriv(), to avoid these when not needed?
  const real dxlogrho2 = 2*x[0]/rho2, dxlogsigma = (4*x[0]*(r2+a2)-2*(x[0]-bhm)*a2*sint2)/sigma, dxlogzz = 1./x[0]-dxlogrho2;
  const real dylogrho2 =-2*a2*sint*cost/rho2, dylogsigma = -2*a2*delta*sint*cost/sigma, dylogzz =-dylogrho2;
#endif
    // --- These functions are expected to be user-provided.
    // Metrics: individual elements (SYCL only for now)
    SYCL_EXTERNAL real alpha();                                   // Time element of the metric. Will put many #ifdef cases.
    SYCL_EXTERNAL real betai(unsigned short i);                   // Each mixed element of the metric
    SYCL_EXTERNAL real gCon (unsigned short i, unsigned short j); // Each element of 3D (spatial) metric
    SYCL_EXTERNAL real gCov (unsigned short i, unsigned short j); // Each element of 3D (spatial) metric
    SYCL_EXTERNAL real gDet (), gDet1 ();                         // For notation
    // Derivatives
    SYCL_EXTERNAL real dgAlpha(unsigned short i);
    SYCL_EXTERNAL real dgBeta (unsigned short i, unsigned short j);
    SYCL_EXTERNAL real dgCov  (unsigned short i, unsigned short j, unsigned short k);
    // END These functions are expected to be user-provided.

    // --- These functions are convenience functions, no modifications should be needed here.
    SYCL_EXTERNAL void  beta  (real bet[NDIM]);
    SYCL_EXTERNAL real g3DCon(real g[9]);
    SYCL_EXTERNAL real g3DCov(real g[9]);
    SYCL_EXTERNAL void con2Cov(real vCon[3], real vCov[3]); // Spatial 3-vectors (fixed for physics)
    SYCL_EXTERNAL void cov2Con(real vCov[3], real vCon[3]); // Spatial 3-vectors (fixed for physics)
    // Maybe not needed?
    SYCL_EXTERNAL void  g4DCon(real g[16]);
    SYCL_EXTERNAL void  g4DCov(real g[16]);
    // END These functions are convenience functions, no modifications should be needed here.
};

#if METRIC == KERR_SCHILD
inline real Metric::bhm = static_cast<real>(1);
inline real Metric::bha = static_cast<real>(0);
inline real Metric::bhc = static_cast<real>(0);
#endif

// --- Inline metric implementations (cross-TU inlining on host, device compilation from header)
#if METRIC == CARTESIAN
inline real Metric::gDet   (){ return 1.0; }
inline real Metric::gDet1  (){ return 1.0; }
inline real Metric::alpha  (){ return 1.0; }
inline real Metric::betai  (unsigned short i){ return 0; }
inline real Metric::gCon   (unsigned short i, unsigned short j){ return (i==j) ? 1.0 : 0.0; }
inline real Metric::gCov   (unsigned short i, unsigned short j){ return (i==j) ? 1.0 : 0.0; }
inline real Metric::dgAlpha(unsigned short i) { return 0.0; }
inline real Metric::dgBeta (unsigned short i, unsigned short j) { return 0.0; }
inline real Metric::dgCov  (unsigned short i, unsigned short j, unsigned short k) { return 0.0; }
#elif METRIC == KERR_SCHILD
inline real Metric::gDet   (){ return sycl::sqrt(rho2*det*sint2); }
inline real Metric::gDet1  (){ return (sint<=1.e-6) ? 0.0 : sycl::rsqrt(rho2*det*sint2); }
inline real Metric::alpha  (){ return sycl::rsqrt(1.+zz); }
inline real Metric::betai  (unsigned short i){ return (0==i) ? (zz/(1.+zz)) : 0.0; }
inline real Metric::gCon   (unsigned short i, unsigned short j){
  switch(i*10+j){
    case  0: return 1.0 + zz;
    case 11: return rho2;
    case 22: return (sigma/rho2)*sint2;
    case  2: case 20: return -a*(1.+zz)*sint2;
    default: return 0.0;
  }
}
inline real Metric::gCov   (unsigned short i, unsigned short j){
  switch(i*10+j){
    case  0: return (sigma/rho2)/det;
    case 11: return    1.0/rho2;
    case 22: return (sint<=1.e-6) ? 0.0 : (1.0+zz)/(det*sint2);
    case  2: case 20: return a*(1.0+zz)/det;
    default: return 0.0;
  }
}
inline real Metric::dgAlpha(unsigned short i) {
  switch(i){
    case  0: return  alpha()*.5*zz/(1.0+zz)*dxlogzz;
    case  1: return -alpha()*.5*zz/(1.0+zz)*dylogzz;
    default: return  0.0;
  }
}
inline real Metric::dgBeta (unsigned short i, unsigned short j) {
  switch(i*10+j){
    case  0: return  zz/(1+zz)/(1.+zz)*dxlogzz;
    case  1: return  zz/(1+zz)/(1.+zz)*dylogzz;
    default: return  0.0;
  }
}
inline real Metric::dgCov  (unsigned short i, unsigned short j, unsigned short k) {
  switch(i*100+j*10+k){
    case   0: return  zz*dxlogzz;
    case 110: return  gCov(1,1)* dxlogrho2;
    case 220: return  gCov(2,2)*(dxlogsigma-dxlogrho2);
    case  20: case 200: return  gCov(0,2)*zz/(1+zz)*dxlogzz;
    case   1: return zz*dylogzz;
    case 111: return gCov(1,1) * dylogrho2;
    case 221: return gCov(2,2) *(dylogsigma-dylogrho2+2*cost/sint);
    case  21: case 201: return gCov(0,2)*(zz/(1.0+zz)*dylogzz+2*cost/sint);
    default : return  0.0;
  }
}
#endif

// --- Inline convenience functions (cross-TU inlining on host, device compilation from header)
inline void Metric::beta(real bet[NDIM]){ bet[0] = betai(0); bet[1] = betai(1); bet[2] = betai(2);}
inline real Metric::g3DCon(real g[9]){
  for (unsigned int ix = 0; ix < 3; ix++)
    for (unsigned int jx = 0; jx < 3; jx++)
      g[ix+3*jx] = gCon(ix, jx);
  return gDet();
}
inline real Metric::g3DCov(real g[9]){
  for (unsigned int ix = 0; ix < 3; ix++)
    for (unsigned int jx = 0; jx < 3; jx++)
      g[ix+3*jx] = gCov(ix, jx);
  return 1.0 / gDet();
}
inline void Metric::con2Cov(real vCon[3], real vCov[3]) {
  vCov[0] = gCov(0, 0) * vCon[0] + gCov(0, 1) * vCon[1] + gCov(0, 2) * vCon[2];
  vCov[1] = gCov(1, 0) * vCon[0] + gCov(1, 1) * vCon[1] + gCov(1, 2) * vCon[2];
  vCov[2] = gCov(2, 0) * vCon[0] + gCov(2, 1) * vCon[1] + gCov(2, 2) * vCon[2];
}
inline void Metric::cov2Con(real vCov[3], real vCon[3]) {
  vCon[0] = gCon(0, 0) * vCov[0] + gCon(0, 1) * vCov[1] + gCon(0, 2) * vCov[2];
  vCon[1] = gCon(1, 0) * vCov[0] + gCon(1, 1) * vCov[1] + gCon(1, 2) * vCov[2];
  vCon[2] = gCon(2, 0) * vCov[0] + gCon(2, 1) * vCov[1] + gCon(2, 2) * vCov[2];
}

#endif
