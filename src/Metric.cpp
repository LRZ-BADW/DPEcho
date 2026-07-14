//   Copyright(C) 2022 Alexander Pöppl, Intel Corporation
//   Copyright(C) 2022 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.
#include "Metric.hpp"
#include "Physics.hpp"
#include "echo.hpp"

// DEAD STATIC FUNC: beta, g3DCon, g3DCov, con2Cov, cov2Con — moved to Metric.hpp as inline
// FIXME: The following two likely need some corrections... (betaCon? betaCov?)
// Presumably, they won't be used, so it doesn't matter.
SYCL_EXTERNAL void Metric::g4DCov(real g[16]){
  real betaCon[3], betaCov[3];
  beta(betaCon);
  con2Cov(betaCon, betaCov);
  g[0] = dot(betaCon, betaCov) - alpha();
  for (int ix = 1; ix < 4; ix++) {
    g[ix] = g[4*ix] = betaCov[ix - 1];
    for (int jx = ix; jx < 4; jx++) {
      g[jx+4*ix] = g[ix + 4*jx] = gCov(ix - 1, jx - 1);
    }
  }
}
SYCL_EXTERNAL void Metric::g4DCon(real g[16]){
  real betaCon[3], betaCov[3];
  beta(betaCon);
  con2Cov(betaCon, betaCov);
  g[0] = dot(betaCon, betaCov) - alpha();
  for (int ix = 1; ix < 4; ix++) {
    g[ix] = g[4*ix] = betaCon[ix - 1];
    for (int jx = ix; jx < 4; jx++) {
      g[jx+4*ix] = g[ix + 4*jx] = gCon(ix - 1, jx - 1);
    }
  }
}
