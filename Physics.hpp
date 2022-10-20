//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Physics_hpp_
#define _Physics_hpp_

#include "echo.hpp"
#include "Logger.hpp"
#include "Metric.hpp"
#include "utils/tb-types.hpp"

#include "echoSycl.hpp"


//-- 3-vectors in GR
SYCL_EXTERNAL void  matMul(field [9], field *, field *    , id<1> =0, unsigned =1);
SYCL_EXTERNAL field dot   (field *u , field *, unsigned =3, id<1> =0, unsigned =1);
SYCL_EXTERNAL void  cross (field *v , field *, field *    , id<1> =0, unsigned =1);

//-- Fields: prim&cons
SYCL_EXTERNAL void prim2cons(id<1> gid, unsigned n, field_array v, field_array u, Metric &m);
SYCL_EXTERNAL void cons2prim(id<1> gid, unsigned n, field_array u, field_array v, Metric &m);

//-- Fluxes: here as they know about the metric
SYCL_EXTERNAL void physicalFlux(int dir, Metric &g, field vD[FLD_TOT], field uD[FLD_TOT], field f[FLD_TOT], field vf[2], field vt[2]);
#endif
