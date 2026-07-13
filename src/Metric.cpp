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

// --- These functions are convenience functions, no modifications should be needed here.
SYCL_EXTERNAL void   Metric::beta(real bet[NDIM]){   bet[0] = betai(0); bet[1] = betai(1);  bet[2] = betai(2);}
SYCL_EXTERNAL real  Metric::g3DCon(real g[9]){
  for (unsigned int ix = 0; ix < 3; ix++)
    for (unsigned int jx = 0; jx < 3; jx++)
      g[ix+3*jx] = gCon(ix, jx);
  return gDet(); // Determinant
}
SYCL_EXTERNAL real Metric::g3DCov(real g[9]){
  for (unsigned int ix = 0; ix < 3; ix++)
    for (unsigned int jx = 0; jx < 3; jx++)
      g[ix+3*jx] = gCov(ix, jx);
  return 1.0 / gDet();
}
SYCL_EXTERNAL void Metric::con2Cov(real vCon[3], real vCov[3]) {
  vCov[0] = gCov(0, 0) * vCon[0] + gCov(0, 1) * vCon[1] + gCov(0, 2) * vCon[2];
  vCov[1] = gCov(1, 0) * vCon[0] + gCov(1, 1) * vCon[1] + gCov(1, 2) * vCon[2];
  vCov[2] = gCov(2, 0) * vCon[0] + gCov(2, 1) * vCon[1] + gCov(2, 2) * vCon[2];
}
SYCL_EXTERNAL void Metric::cov2Con(real vCov[3], real vCon[3]) {
  vCon[0] = gCon(0, 0) * vCov[0] + gCon(0, 1) * vCov[1] + gCon(0, 2) * vCov[2];
  vCon[1] = gCon(1, 0) * vCov[0] + gCon(1, 1) * vCov[1] + gCon(1, 2) * vCov[2];
  vCon[2] = gCon(2, 0) * vCov[0] + gCon(2, 1) * vCov[1] + gCon(2, 2) * vCov[2];
}
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
