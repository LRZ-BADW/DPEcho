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
#define hipSYCL   3

#if   SYCL == hipSYCL
namespace mysycl = hipsycl::sycl;
namespace my1api = hipsycl::sycl;
#define SYCL_EXTERNAL // Not needed but interfaces must be made uniform
#elif SYCL == LLVM
namespace mysycl = cl::sycl;
namespace my1api = cl::sycl::ext::oneapi;
#elif SYCL == oneAPIold
namespace mysycl = cl::sycl;
namespace my1api = cl::sycl::ONEAPI;
#else
namespace mysycl = cl::sycl;
namespace my1api = cl::sycl::ext::oneapi;
#endif

using namespace mysycl;
using namespace my1api;

//-- To use prinft within device code
#ifdef __SYCL_DEVICE_ONLY__
  #define CONSTANT __attribute__((opencl_constant))
#else
  #define CONSTANT
#endif

#endif
