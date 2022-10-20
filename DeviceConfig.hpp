//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//   Copyright(C) 2022 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _DeviceConfig_hpp_
#define _DeviceConfig_hpp_

#pragma once

#include "echoSycl.hpp"

struct DeviceConfig {
  private:
  std::vector<device> devices;

  public:
  DeviceConfig();
  void printTargetInfo(mysycl::queue);
  void listDevices();
  device deviceWith(int id);
  device debugDevice();
  std::vector<device> gpus();
  std::vector<device> cpus();
};

#endif
