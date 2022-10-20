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

#include "echo.hpp"
#include "Logger.hpp"
#include "utils/tb-types.hpp"

#include "echoSycl.hpp"

class Metric {
  private:
    field coords[3];

  public:
    Metric (field x1, field x2, field x3) { coords[0] = x1; coords[1] = x2; coords[2] = x3; }

#if METRIC > CARTESIAN
    static field bhm = 0.0, bha = 0.0,  bhc = 0.0;
    // read them from file on host (in main) and set here once...
    static inline void setParameters(field bhm, field bha, field bhc) {
      Metric::bhm = bhm;    Metric::bha = bha;    Metric::bhc = bhc;
    }
#else // For uniformity
    static inline void setParameters(field bhm, field bha, field bhc) {}
#endif

#if METRIC == CARTESIAN
    // Nothing needed
#elif METRIC == KS
    // declare needed vars
#endif

    // --- These functions are expected to be user-provided.
    // Metrics: individual elements (SYCL only for now)
    SYCL_EXTERNAL field alpha();                               // Time element of the metric. Will put many #ifdef cases.
    SYCL_EXTERNAL field betai(unsigned int i);                 // Each mixed element of the metric
    SYCL_EXTERNAL field gCon (unsigned int i, unsigned int j); // Each element of 3D (spatial) metric
    SYCL_EXTERNAL field gCov (unsigned int i, unsigned int j); // Each element of 3D (spatial) metric
    SYCL_EXTERNAL field gDet ();
    // Derivatives
    SYCL_EXTERNAL field dgAlpha(field i);
    SYCL_EXTERNAL field dgBeta (field i, field j);
    SYCL_EXTERNAL field dgCov  (field i, field j, field k);
    // --- These functions are expected to be user-provided.

    // --- These functions are convenience functions, no modifications should be needed here.
    SYCL_EXTERNAL void  beta  (field bet[3]);
    SYCL_EXTERNAL field g3DCon(field g[9]);
    SYCL_EXTERNAL field g3DCov(field g[9]);
    SYCL_EXTERNAL void con2Cov(field vCon[3], field vCov[3]);
    SYCL_EXTERNAL void cov2Con(field vCov[3], field vCon[3]);
    // Maybe not needed?
    SYCL_EXTERNAL void  g4DCon(field g[16]);
    SYCL_EXTERNAL void  g4DCov(field g[16]);
    // --- These functions are convenience functions, no modifications should be needed here.
};

#endif
