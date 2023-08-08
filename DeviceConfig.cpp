//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//   Copyright(C) 2022 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "DeviceConfig.hpp"
#include "Logger.hpp"
#include <sstream>

DeviceConfig::DeviceConfig() {
  for (auto const &p : mysycl::platform::get_platforms())
    for ( auto dev : p.get_devices() )
      devices.push_back(dev);
}

device DeviceConfig::deviceWith(int id){
  Logger *log = Logger::getInstance(); log->setPar(true);
  device temp;
  if( (id>=0) && (id<devices.size()) ){
    ((*log)+2)<<TAG <<"Looking at device #"<<id; log->fl();
    temp = devices[id];
  }else{
    ((*log)+2)<<TAG <<"Looking at DPEcho default device."; log->fl();
    listDevices();
#if   SYCL==oneAPI || SYCL==LLVM || SYCL==OpenSYCL
#if   DEVICE==DEV_CPU
    temp = mysycl::device(mysycl::cpu_selector_v); // mysycl::cpu_selector_v         fallBackSel;
#elif DEVICE==DEV_GPU
    temp = mysycl::device(mysycl::gpu_selector_v); // mysycl::gpu_selector_v         fallBackSel;
#elif DEVICE==DEV_ACC
    temp = mysycl::device(mysycl::accelerator_selector_v); // mysycl::accelerator_selector_v fallBackSel;
#elif DEVICE==DEV_FPGA
    temp = mysycl::device(mysycl::accelerator_selector_v); //    mysycl::accelerator_selector_v fallBackSel;
#else // host is deprecated in SYCL2020
    temp = mysycl::device(mysycl::default_selector_v); //    mysycl::default_selector_v     fallBackSel;
#endif
#else
#if   DEVICE==DEV_CPU
    mysycl::cpu_selector         fallBackSel;
#elif DEVICE==DEV_GPU
    mysycl::gpu_selector         fallBackSel;
#elif DEVICE==DEV_ACC
    mysycl::accelerator_selector fallBackSel;
#elif DEVICE==DEV_FPGA
    mysycl::accelerator_selector fallBackSel;
#else // host is deprecated in SYCL2020
    mysycl::default_selector     fallBackSel;
#endif
    temp = mysycl::device(fallBackSel);
#endif
  }
  printTargetInfo(temp);
  return temp;
}

void DeviceConfig::listDevices() {
  Logger *log = Logger::getInstance(); log->setPar(false);
  ((*log) + 2) << TAG <<"\n\t# Available SYCL devices:\t" << devices.size() ;
  for (size_t i = 0; i < devices.size(); i++) {
    bool hasDpSupport = devices[i].has(aspect::fp64);
    (*log) <<"\n\t- Device #" << i << ":\t"
      << devices[i].get_info<info::device::name>() << " ("
      << devices[i].get_info<info::device::max_compute_units>() << " EUs"
      << (hasDpSupport ? "" : ", SP only") << ")";
  }
  log->fl();
}

std::vector<device> DeviceConfig::gpus() {
  std::vector<device> res;
  for (auto d : this->devices) {
    if (d.has(aspect::gpu)) {
      res.push_back(d);
    }
  }
  return res;
}

std::vector<device> DeviceConfig::cpus() {
  std::vector<device> res;
  for (auto d : this->devices) {
    if (d.has(aspect::cpu)) {
      res.push_back(d);
    }
  }
  return res;
}

device DeviceConfig::debugDevice() {
  std::vector<device> res;
  for (auto d : this->devices) {
    if (d.get_info<info::device::name>().find("Host") != -1 || d.get_info<info::device::name>().find("host") != -1) {
      return d;
    }
  }
  throw std::runtime_error("No debug device is available on this machine!");
}

void  DeviceConfig::printTargetInfo(device dev) {
  Logger *Log = Logger::getInstance(); Log->setPar(true);
  *Log+0<<TAG
        << "\n\tHardware "   << dev.get_info<info::device::name>() // << " is " << (dev.is_host()? "HOST ":"")
        << (dev.is_cpu()? "CPU ":"") << (dev.is_gpu()? "GPU ":"") << (dev.is_accelerator()? " ACCELERATOR ":"")
        << "\n\tMax Compute Units  : " << dev.get_info<info::device::max_compute_units>  ();   Log->fl();
#if SYCL<=oneAPI
  *Log+0<< "\n\tMax Work Group Size: " << dev.get_info<info::device::max_work_group_size>()
        << "\n\tGlobal Memory / GB : " << dev.get_info<info::device::global_mem_size>    ()/pow(1024.0, 3)
        << "\n\tLocal  Memory / kB : " << dev.get_info<info::device::local_mem_size>     ()/1024.0      ;  Log->fl();
#else
  *Log+18<<"\n\tMax Work Group Size, Global and Local Memory queries are handled differently outside oneAPI.";
#endif
  Log->fl();
}
