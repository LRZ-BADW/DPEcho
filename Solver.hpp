//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Solver_hpp_
#define _Solver_hpp_

#include "echo.hpp"
#include "utils/tb-types.hpp"

#include "echoSycl.hpp"

#if NRK==1
const float crk1[] = {0.0};
const float crk2[] = {1.0};
#elif NRK==2
const float crk1[] = {0.0, 0.5};
const float crk2[] = {1.0, 0.5};
#elif NRK==3
const float crk1[] = {0.0, 0.75, 1.0/3.0};
const float crk2[] = {1.0, 0.25, 2.0/3.0};
#endif

// Only those called directly outside Solver.cpp needed.
SYCL_EXTERNAL field holibDer(int myId, field *var, int stride);
SYCL_EXTERNAL void  holibRec(int myId, field *var, int stride,  field *vL, field *vR); // Reconstructs at i +/- 1/2

#endif
