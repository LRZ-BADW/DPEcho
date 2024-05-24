//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//   Copyright(C) 2022 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Device_hpp_
#define _Device_hpp_

#pragma once

#include "Parameters.hpp"

#include <sycl/sycl.hpp>

//- Device  Selection
#define DEV_DEF  0
#define DEV_CPU  1
#define DEV_GPU  2
#define DEV_ACC  3

struct Device {
  private:
    std::vector<sycl::device> devices;

  public:
    Device();
    void printTargetInfo(sycl::device);
    void listDevices();
    sycl::device deviceWith(Parameters &p);
    sycl::device debugDevice();
    std::vector<sycl::device> gpus();
    std::vector<sycl::device> cpus();
};

#endif
