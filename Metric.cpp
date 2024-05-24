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

#if   METRIC == CARTESIAN  // We will put #ifdef for the different metrics (ALL analytic here!)
SYCL_EXTERNAL field Metric::gDet   (){ return 1.0; }
SYCL_EXTERNAL field Metric::gDet1  (){ return 1.0; }
SYCL_EXTERNAL field Metric::alpha  (){ return 1.0; }
SYCL_EXTERNAL field Metric::betai  (unsigned short i){ return 0;}  // Controvariant beta element of the metric
SYCL_EXTERNAL field Metric::gCon   (unsigned short i, unsigned short j){ return (i==j) ? 1.0 : 0.0; }
SYCL_EXTERNAL field Metric::gCov   (unsigned short i, unsigned short j){ return (i==j) ? 1.0 : 0.0; }
SYCL_EXTERNAL field Metric::dgAlpha(unsigned short i) {  return 0.0; } // Derivatives
SYCL_EXTERNAL field Metric::dgBeta (unsigned short i, unsigned short j) {  return 0.0; }
SYCL_EXTERNAL field Metric::dgCov  (unsigned short i, unsigned short j, unsigned short k) { return 0.0; }
#elif METRIC == KERR_SCHILD
// WARNING: Mostly untested!
SYCL_EXTERNAL field Metric::gDet   (){ return sycl::sqrt(rho2*det*sint2); }
SYCL_EXTERNAL field Metric::gDet1  (){ return (sint<=1.e-6) ? 0.0 : sycl::rsqrt(rho2*det*sint2) ; }
SYCL_EXTERNAL field Metric::alpha  (){ return sycl::rsqrt(1.+zz); }
SYCL_EXTERNAL field Metric::betai  (unsigned short i){ return (0==i) ? (zz/(1.+zz)) : 0.0; }
SYCL_EXTERNAL field Metric::gCon   (unsigned short i, unsigned short j){
  switch(i*10+j){
    case  0: return 1.0 + zz;
    case 11: return rho2;
    case 22: return (sigma/rho2)*sint2;
    case  2: case 20: return -a*(1.+zz)*sint2;
    default: return 0.0;
  }
}
SYCL_EXTERNAL field Metric::gCov   (unsigned short i, unsigned short j){
  switch(i*10+j){
    case  0: return (sigma/rho2)/det;
    case 11: return    1.0/rho2;
    case 22: return (sint<=1.e-6) ? 0.0 : (1.0+zz)/(det*sint2);
    case  2: case 20: return a*(1.0+zz)/det;
    default: return 0.0;
  }
}
// Derivatives
SYCL_EXTERNAL field Metric::dgAlpha(unsigned short i) {
  switch(i){
    case  0: return  alpha()*.5*zz/(1.0+zz)*dxlogzz;
    case  1: return -alpha()*.5*zz/(1.0+zz)*dylogzz;
    default: return  0.0;
  }
}
SYCL_EXTERNAL field Metric::dgBeta (unsigned short i, unsigned short j) {
  switch(i*10+j){
    case  0: return  zz/(1+zz)/(1.+zz)*dxlogzz;
    case  1: return  zz/(1+zz)/(1.+zz)*dylogzz;
    default: return  0.0;
  }
}
SYCL_EXTERNAL field Metric::dgCov  (unsigned short i, unsigned short j, unsigned short k) {
  switch(i*100+j*10+k){
    case   0: return  zz*dxlogzz; //-- Along r
    case 110: return  gCov(1,1)* dxlogrho2;
    case 220: return  gCov(2,2)*(dxlogsigma-dxlogrho2);
    case  20: case 200: return  gCov(0,2)*zz/(1+zz)*dxlogzz;
    case   1: return zz*dylogzz;  //-- Along theta
    case 111: return gCov(1,1) * dylogrho2;
    case 221: return gCov(2,2) *(dylogsigma-dylogrho2+2*cost/sint);
    case  21: case 201: return gCov(0,2)*(zz/(1.0+zz)*dylogzz+2*cost/sint);
    default : return  0.0;        //-- Along phi and the others are all 0s
  }
}
#endif // METRIC == TYPE

// --- These functions are convenience functions, no modifications should be needed here.
SYCL_EXTERNAL void   Metric::beta(field bet[3]){   bet[0] = betai(0); bet[1] = betai(1);  bet[2] = betai(2);}
SYCL_EXTERNAL field  Metric::g3DCon(field g[9]){
  for (unsigned int ix = 0; ix < 3; ix++)
    for (unsigned int jx = 0; jx < 3; jx++)
      g[ix+3*jx] = gCon(ix, jx);
  return gDet(); // Determinant
}
SYCL_EXTERNAL field Metric::g3DCov(field g[9]){
  for (unsigned int ix = 0; ix < 3; ix++)
    for (unsigned int jx = 0; jx < 3; jx++)
      g[ix+3*jx] = gCov(ix, jx);
  return 1.0 / gDet();
}
SYCL_EXTERNAL void Metric::con2Cov(field vCon[3], field vCov[3]) {
  vCov[0] = gCov(0, 0) * vCon[0] + gCov(0, 1) * vCon[1] + gCov(0, 2) * vCon[2];
  vCov[1] = gCov(1, 0) * vCon[0] + gCov(1, 1) * vCon[1] + gCov(1, 2) * vCon[2];
  vCov[2] = gCov(2, 0) * vCon[0] + gCov(2, 1) * vCon[1] + gCov(2, 2) * vCon[2];
}
SYCL_EXTERNAL void Metric::cov2Con(field vCov[3], field vCon[3]) {
  vCon[0] = gCon(0, 0) * vCov[0] + gCon(0, 1) * vCov[1] + gCon(0, 2) * vCov[2];
  vCon[1] = gCon(1, 0) * vCov[0] + gCon(1, 1) * vCov[1] + gCon(1, 2) * vCov[2];
  vCon[2] = gCon(2, 0) * vCov[0] + gCon(2, 1) * vCov[1] + gCon(2, 2) * vCov[2];
}
// FIXME: The following two likely need some corrections... (betaCon? betaCov?)
// Presumably, they won't be used, so it doesn't matter.
SYCL_EXTERNAL void Metric::g4DCov(field g[16]){
  field betaCon[3], betaCov[3];
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
SYCL_EXTERNAL void Metric::g4DCon(field g[16]){
  field betaCon[3], betaCov[3];
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
