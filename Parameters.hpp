//  Copyright(C) 2023 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
//  with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is
//  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and limitations under the License.

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#pragma once

#include "utils/tb-types.hpp"

#include <iostream>
#include <type_traits>
#include <string>
#include <unordered_map>

class Parameters {

    std::unordered_map<std::string, std::string> parameters;
    std::unordered_map<std::string, int> numUses;

  public:

    Parameters(std::string &filename);

    template<typename T> T getOr(std::string key, T defaultValue, T (*convert)(std::string &), bool warnOnDefault = false) {
      if (has(key)) {
        return convert(parameters[key]);
      }
      if (warnOnDefault) std::cout << "WARN: using default value " << defaultValue << " for key " << key << std::endl;
      return defaultValue;
    }

    template<typename T> T getOr(std::string key, T defaultValue, bool warnOnDefault = false) {
      numUses[key] += 1;
      if (has(key)) {
        if constexpr (std::is_same_v<int, T>) {
          return std::stoi(parameters[key]);
        } else if constexpr (std::is_same_v<field, T>) {
          return static_cast<field>(std::stod(parameters[key]));
        } else if constexpr (std::is_convertible_v<T, std::string>) {
          return parameters[key];
        }
      }
      if (warnOnDefault) std::cout << "WARN: using default value " << defaultValue << " for key " << key << std::endl;
      return defaultValue;
    }

    void report();
    bool has(std::string &key);
};

#endif
