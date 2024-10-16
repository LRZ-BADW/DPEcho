//   Copyright(C) 2021 Alexander Pöppl, Intel Corp.
//   Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef DEBUG_OUTPUT_HPP
#define DEBUG_OUTPUT_HPP

#include "utils/tb-types.hpp"
#include "Grid.hpp"

#include <iostream>
#include <string>

#pragma once

class Problem;

namespace output {
  void writeArray(Problem &problem, Grid &gr, std::string dir="out", std::string name="task");
}

#endif
