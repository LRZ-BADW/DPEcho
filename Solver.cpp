//  Copyright(C) 2021 Salvatore Cielo, LRZ
//  Copyright(C) 2021 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Solver.hpp"

#include "Physics.hpp"
#include "Metric.hpp"


//-- Derivation from staggered f_{i-ngc:i+ngc-1} to cell-centered hat f_{i}
SYCL_EXTERNAL field holibDer(int myId, field *var, int stride){
#ifndef FD // Same as FD==2
  return var[myId]-var[myId-stride];
#else
#if   FD==2 //-- 2nd order derivation (ngc=1 required)
  const field d0  = -1.0, d1 = 1.0;
  const field d[] = {d0, d1};
#elif FD==4 //-- 4th order derivation (ngc=2 required)
  const field d0  = 13.0/12.0, d2 = -1.0/24.;
  const field d[] = {-d2,d2-d0,d0-d2,d2};
#elif FD==6 //-- 6th order derivation (ngc=3 required)
  const field d0 = 1067/960., d2=-29/480., d4=3/640.;
  const field d[]={-d4,d4-d2,d2-d0,d0-d2,d2-d4,d4};
#endif // on FD values
  field  sum=0.0;
  for(int i=0; i<FD;++i){ sum+= d[i] * var[myId+stride*(i-FD/2)];}
  return sum;
#endif // ifndef FD
}

//-- Reconstruction Filters

SYCL_EXTERNAL inline field holibPhi(field r){ //-- rec of echo-mini, may be dropped
#if RECONSTR==NO
  return 0.0;
#elif RECONSTR==LINEAR
  return 1.0; // Linear, mostly for debug
#elif RECONSTR==MINMOD
  return mysycl::max(0.0, mysycl::min(1.0, r)); // minmod
#elif RECONSTR==MONCEN
  field tmin = mysycl::min(2.0*r, 0.5*(1.0 + r));
  return mysycl::max(0.0, mysycl::min(tmin, 2.0)); // monotonized central
#elif RECONSTR==VANLEER
  return (r + abs(r))/(1. + abs(r));  // van Leer
#else
  return 0.0;
#endif
}

SYCL_EXTERNAL inline field mm2(field d1, field d2){
  return (d1*d2 <=0)? 0.0 : sign(d1) * mysycl::fmin(mysycl::fabs(d1), mysycl::fabs(d2));
}

SYCL_EXTERNAL inline field mc2(field d1, field d2) {
  field  coeff = ((d1*d2 <=0)? 0.0 : sign(d1));
  return coeff * mysycl::fmin(2*mysycl::fabs(d1), mysycl::fmin(2*mysycl::fabs(d2), 0.5*mysycl::fabs(d1+d2)));
}

SYCL_EXTERNAL inline field mm4(field d1, field d2, field d3, field d4) {
  field s1 = sign(d1), s2 = sign(d2), s3 = sign(d3), s4 = sign(d4);
  field signCalc = (s1+s2)* mysycl::fabs((s1 + s3) * (s1 + s4));

  return 0.125 * signCalc * mysycl::fmin( mysycl::fmin(mysycl::fabs(d1), mysycl::fabs(d2)),
					  mysycl::fmin(mysycl::fabs(d3), mysycl::fabs(d4)) );
}

SYCL_EXTERNAL inline field mp5(field f, field f1, field f2, field f3, field f4, field f5) {
//  See also: Pizzarelli, Marco and Ahn, Myeong-Hwan and Lee, Duck-Joo, 2019
//    Hybrid Flux Method in Monotonicity-Preserving Scheme for Accurate and Robust Simulation in Supersonic Flow
//    https://doi.org/10.1155/2019/4590956
  field ful, fmp, dm, d0, dp, fmd, flc, myfmin, myfmax;
  field res = f;

  ful = f3 + 2 * (f3 - f2);
  fmp = f3 + mm2(f4 - f3, ful - f3);
  if ((f - f3) * (f - fmp) <= 0.0) {
    return f;
  } else {
    dm = f1-2*f2+f3;
    d0 = f2-2*f3+f4;
    dp = f3-2*f4+f5;
    fmd = .5*(f3+f4)-.5*mm4(4*d0-dp,4*dp-d0,d0,dp);
    flc = f3+.5*(f3-f2)+(4./3.)*mm4(4*d0-dm,4*dm-d0,d0,dm);
    myfmin = mysycl::fmax( mysycl::fmin(f3, mysycl::fmin(f4,fmd)), mysycl::fmin(f3, mysycl::fmin(ful,flc)) );
    myfmax = mysycl::fmin( mysycl::fmax(mysycl::fmax(f3,f4), fmd), mysycl::fmax(mysycl::fmax(f3,ful), flc) );
    return f + mm2(myfmin-f, myfmax-f);
  }
}

//-- Reconstruction at (i +/- 1/2)
SYCL_EXTERNAL void holibRec(int myId, field *var, int offset, field *vL, field *vR){
#if REC_ORDER == 1
  field  r, a0, a1, a2, a3;
  a0 = var[myId-offset],   a1 = var[myId];
  a2 = var[myId+offset],   a3 = var[myId+2*offset];

  r = (a2 - a3)/(a1 - a2 + 1.e-12);  *vR = a2 + 0.5*(a1-a2) * holibPhi(r); // Together, for cache reuse!
  r = (a1 - a0)/(a2 - a1 + 1.e-12);  *vL = a1 + 0.5*(a2-a1) * holibPhi(r); // rec of echo-mini. May be discarded; it is usable.
#elif REC_ORDER == 2
  field fBuf[REC_TOTAL_POINTS];
  for (int i = 0; i < REC_TOTAL_POINTS; i++) {
    fBuf[i] = var[myId + offset * (i - REC_LEFT_OFFSET)];
  }
  *vL = fBuf[1] + 0.5 * mm2(fBuf[1] - fBuf[0], fBuf[2] - fBuf[1]);
  *vR = fBuf[2] + 0.5 * mm2(fBuf[2] - fBuf[3], fBuf[1] - fBuf[2]);
#elif REC_ORDER == 5
  field fBuf[REC_TOTAL_POINTS], f[2] = {0.0, 0.0};
  field const d[5] = {3./128., -20./128., 90./128., 60./128., -5./128.};

  for (int i = 0; i < REC_TOTAL_POINTS; i++)
    fBuf[i] = var[myId + offset * (i - REC_LEFT_OFFSET)];

  for (int i = 0; i < 5; i++){ f[0] += d[i] * fBuf[  i]; }
  for (int i = 0; i < 5; i++){ f[1] += d[i] * fBuf[5-i]; }

  *vL = mp5(f[0], fBuf[0], fBuf[1], fBuf[2], fBuf[3], fBuf[4]);
  *vR = mp5(f[1], fBuf[5], fBuf[4], fBuf[3], fBuf[2], fBuf[1]);
#else
#error "Unsupported value for the order of reconstruction"
#endif
}
