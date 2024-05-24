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
    [[maybe_unused]] field x[3];

  public:
    Metric (field x0, field x1, field x2) : x{x0, x1, x2 } {} // Initialize x directly upon construction. Theory: not doing so causes issues with computations below.
#if METRIC > CARTESIAN
    static field bhm, bha, bhc;
    // read them from file on host (in main) and set here once...
    static inline void setParameters(field mm, field aa, field cc) {
      Metric::bhm = mm; Metric::bha = aa; Metric::bhc = cc;
#if METRIC == KERR_SCHILD
#warning "The Kerr-Schild Metric is not tested (may be incorrect!) and has no proper Problem case. It is expected to fail!"
      Log::cout(4) << TAG << "WARNING: The Kerr-Schild Metric is not tested (may be incorrect!) and has no proper Problem case. It is expected to fail!" << Log::endl;
#endif
    }
#else // For uniformity
    static inline void setParameters(field bhm, field bha, field bhc) {}
#endif

#if   METRIC == CARTESIAN   // Nothing needed
#elif METRIC == KERR_SCHILD // declare needed vars
  // WOW!!!
  const field r2 = x[0]*x[0], sint = sycl::sin(x[1]), sint2 = sint*sint, cost = sycl::cos(x[1]);
  const field a = bha, a2 = a*a, delta= r2-2.*bhm*x[0]+a2, rho2 = r2+a2*(1.-sint2), zz = 2.*bhm*x[0]/rho2;
  const field sigma = (r2+a2)*(r2+a2)-a2*delta*sint2, det = (1.+zz)*(sigma/rho2-a2*(1.+zz)*sint2);
  // For derivatives TODO: maybe make these into a function Metric::initDeriv(), to avoid these when not needed?
  const field dxlogrho2 = 2*x[0]/rho2, dxlogsigma = (4*x[0]*(r2+a2)-2*(x[0]-bhm)*a2*sint2)/sigma, dxlogzz = 1./x[0]-dxlogrho2;
  const field dylogrho2 =-2*a2*sint*cost/rho2, dylogsigma = -2*a2*delta*sint*cost/sigma, dylogzz =-dylogrho2;
#endif
    // --- These functions are expected to be user-provided.
    // Metrics: individual elements (SYCL only for now)
    SYCL_EXTERNAL field alpha();                                   // Time element of the metric. Will put many #ifdef cases.
    SYCL_EXTERNAL field betai(unsigned short i);                   // Each mixed element of the metric
    SYCL_EXTERNAL field gCon (unsigned short i, unsigned short j); // Each element of 3D (spatial) metric
    SYCL_EXTERNAL field gCov (unsigned short i, unsigned short j); // Each element of 3D (spatial) metric
    SYCL_EXTERNAL field gDet (), gDet1 ();                         // For notation
    // Derivatives
    SYCL_EXTERNAL field dgAlpha(unsigned short i);
    SYCL_EXTERNAL field dgBeta (unsigned short i, unsigned short j);
    SYCL_EXTERNAL field dgCov  (unsigned short i, unsigned short j, unsigned short k);
    // END These functions are expected to be user-provided.

    // --- These functions are convenience functions, no modifications should be needed here.
    SYCL_EXTERNAL void  beta  (field bet[3]);
    SYCL_EXTERNAL field g3DCon(field g[9]);
    SYCL_EXTERNAL field g3DCov(field g[9]);
    SYCL_EXTERNAL void con2Cov(field vCon[3], field vCov[3]);
    SYCL_EXTERNAL void cov2Con(field vCov[3], field vCon[3]);
    // Maybe not needed?
    SYCL_EXTERNAL void  g4DCon(field g[16]);
    SYCL_EXTERNAL void  g4DCov(field g[16]);
    // END These functions are convenience functions, no modifications should be needed here.
};

#if METRIC == KERR_SCHILD
inline field Metric::bhm = static_cast<field>(1);
inline field Metric::bha = static_cast<field>(0);
inline field Metric::bhc = static_cast<field>(0);
#endif

#endif
