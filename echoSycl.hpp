//   Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _echoSycl_hpp_
#define _echoSycl_hpp_

#include <CL/sycl.hpp>

#define oneAPIold 0
#define oneAPI    1
#define LLVM      2
#define OpenSYCL  3

#if   SYCL == OpenSYCL
namespace mysycl = hipsycl::sycl;
namespace my1api = hipsycl::sycl;
#if SYCL_ARCH == AMD
// Used to support relocatable device code (RDC) with OpenSYCL
#define SYCL_EXTERNAL __host__ __device__
#else
#define SYCL_EXTERNAL // Not needed but interfaces must be made uniform
#endif
#else
namespace mysycl = cl::sycl;
namespace my1api = cl::sycl::ext::oneapi;
#endif

using namespace mysycl;
using namespace my1api;

//- Device  Selection
#define DEV_DEF  0
#define DEV_CPU  1
#define DEV_GPU  2
#define DEV_ACC  3

//-- To use printf within device code
#ifdef __SYCL_DEVICE_ONLY__
  #define CONSTANT __attribute__((opencl_constant))
#else
  #define CONSTANT
#endif

#endif
