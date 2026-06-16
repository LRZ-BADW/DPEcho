//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//   Copyright(C) 2022 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Device.hpp"
#include "Logger.hpp"
#include <sstream>

using namespace sycl;

Device::Device() {
  for (auto const &p : sycl::platform::get_platforms())
    for ( auto dev : p.get_devices() )
      devices.push_back(dev);
}


#if   DEVICE==DEV_CPU
  #define SELECTOR sycl::cpu_selector_v
#elif DEVICE==DEV_GPU
  #define SELECTOR sycl::gpu_selector_v
#elif DEVICE==DEV_ACC
  #define SELECTOR sycl::accelerator_selector_v
#elif DEVICE==DEV_FPGA
  #define SELECTOR sycl::accelerator_selector_v
#else
  #define SELECTOR sycl::default_selector_v
#endif


device Device::deviceWith(Parameters &p) {
  listDevices();
  std::string choice = p.getOr<std::string>("deviceSelection", "default");
  int deviceOffset = p.getOr("deviceOffset", 0);
  int deviceCount = p.getOr("deviceCount", 1);

  if (choice == "default") {
    device d(SELECTOR);
    printTargetInfo(d);
    return d;
  } else {
    int myRank = Log::mpiRank();
    int localRanks = Log::mpiRanksPerNode();
    int devIdx = (myRank % localRanks) % deviceCount + deviceOffset;
    Log::Assert(devIdx < static_cast<int>(devices.size())
                 || devIdx < 0 
                 || deviceOffset < 0 
                 || deviceCount <= 0, 
                 "Invalid choice of deviceOffset/deviceCount. Check log for device list.");

    Log::clog() << TAG << " Round-Robin Device choice for rank "<< myRank 
                << ": #" << devIdx << Log::endl;
    printTargetInfo(devices[devIdx]);
    return devices[devIdx];
  }
}
#undef SELECTOR

void Device::listDevices() {
  Log::clog(2) << TAG <<"\n# Available SYCL devices:\t" << devices.size() ;
  for (size_t i = 0; i < devices.size(); i++) {
    bool hasDpSupport = devices[i].has(aspect::fp64);
    Log::clog(2) <<"\n - Device #" << i << ":\t"
      << devices[i].get_platform().get_info<info::platform::name>() << " -> "
      << devices[i].get_info<info::device::name>() << " ("
      << devices[i].get_info<info::device::max_compute_units>() << " EUs"
      << (hasDpSupport ? "" : ", SP only") << ")";
  }
  Log::clog(2) << Log::endl;
}

std::vector<device> Device::gpus() {
  std::vector<device> res;
  for (auto d : this->devices) {
    if (d.has(aspect::gpu)) {
      res.push_back(d);
    }
  }
  return res;
}

std::vector<device> Device::cpus() {
  std::vector<device> res;
  for (auto d : this->devices) {
    if (d.has(aspect::cpu)) {
      res.push_back(d);
    }
  }
  return res;
}

device Device::debugDevice() {
  std::vector<device> res;
  for (auto d : this->devices) {
    if (d.get_info<info::device::name>().find("Host") != std::string::npos
     || d.get_info<info::device::name>().find("host") != std::string::npos) {
      return d;
    }
  }
  throw std::runtime_error("No debug device is available on this machine!");
}

void  Device::printTargetInfo(device dev) {
  Log::clog(  )<< TAG
               << "\n\tPlatform "   << dev.get_platform().get_info<info::platform::name>()
               << "\n\tHardware "   << dev.get_info<info::device::name>() << " - "
               << (dev.is_cpu()? "CPU ":"") << (dev.is_gpu()? "GPU ":"") << (dev.is_accelerator()? " ACCELERATOR ":"")
               << "\n\tMax Compute Units  : " << dev.get_info<info::device::max_compute_units>  ();
  // When we last tried AdaptiveCPP, these calls caused issues. However, they are part of SYCL2020.
  Log::clog(  )<< "\n\tMax Work Group Size: " << dev.get_info<info::device::max_work_group_size>()
               << "\n\tGlobal Memory / GB : " << dev.get_info<info::device::global_mem_size>    ()/std::pow(1024.0, 3)
               << "\n\tLocal  Memory / kB : " << dev.get_info<info::device::local_mem_size>     ()/1024.0;
  Log::clog(  )<< Log::endl;
}
