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

#include "Metric.hpp"
#include "utils/tb-types.hpp"
#include "Physics/Common.hpp"
#include "echo.hpp"
#include <sycl/sycl.hpp>

class Physics {
public:
  static constexpr int MHD   = 0;
  static constexpr int GRMHD = 1;
  static constexpr int HD    = 2;
  static constexpr int GRHD  = 3;

  Physics(const std::string &name);
  int type()  const { return type_;  }
  int fldTot() const { return fldTot_; }
  bool isMagnetic() const { return fldTot_ == 8; }
  const char* name() const;

  SYCL_EXTERNAL void prim2cons  (sycl::id<1> myId, unsigned n, real_array v, real_array u, Metric &g);
  SYCL_EXTERNAL void cons2prim  (sycl::id<1> myId, unsigned n, real_array u, real_array v, Metric &g, real tol = 1.e-9);
  SYCL_EXTERNAL void physicalFlux  (int dir, Metric &g, real *vD, real *uD, real *f, real vf[2], real vt[2]);
  SYCL_EXTERNAL void physicalSource(sycl::id<1> myId, real_array v, Metric &g, real src[4]);

private:
  int type_;
  int fldTot_;
};

#endif
