//  Copyright(C) 2023 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
//  with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is
//  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and limitations under the License.

#include "Parameters.hpp"

#include "Logger.hpp"
#include "utils/tb-types.hpp"

#include <fstream>

Parameters::Parameters(std::string &filename) {
  using namespace std::string_literals;
  std::ifstream fileIn(filename);
  if (fileIn.is_open()) {
    std::string line;
    std::string delimiter = " ";
    while (std::getline(fileIn, line)) {
        if (line.rfind("//", 0) == 0) {
#ifndef NDEBUG
          Log::cout(18)<<TAG<< "COMMENT: " << line << Log::endl;
#endif
        } else {
          std::string key     = line.substr(0, line.find(delimiter));
          std::string content = line.erase (0, line.find(delimiter) + delimiter.length());
          parameters[key] = content;
#ifndef NDEBUG
          Log::cout(18) << "\tCONTENT: " << "KEY: " << key << " VALUE: " << content << Log::endl;
#endif
        }
    }
    fileIn.close();
  } else {
    Log::Assert(false, "Parameters File: "s + filename + " is not valid. Aborting."s);
  }
}

void Parameters::report() {

  Log::cout(5) << TAG << "Parameters that were both loaded and used";
  for (auto &me : parameters) {
    if (numUses[me.first] > 0) {
      Log::cout(5) << "\n\t\"" << me.first << "\" -> \"" << me.second << "\" (" << numUses[me.first] <<" uses)";
    }
  }; Log::cout(5) << Log::endl;

  Log::cout(5) << TAG << "Parameters that were loaded and NOT used" ;
  for (auto &me : parameters) {
    if (numUses[me.first] == 0) {
      Log::cout(5) << "\n\t\"" << me.first << "\" -> \"" << me.second << "\"" ;
    }
  }; Log::cout(5) << Log::endl;

  Log::cout(5) << TAG << "Parameters that were used but NOT loaded" ;
  for (auto &me : numUses) {
    if (parameters.find(me.first) == parameters.end()) {
      Log::cout(5) << "\n\t\"" << me.first << "\" was accessed " << me.second << " times." ;
    }
  }; Log::cout(5) << Log::endl;
}

bool Parameters::has(std::string &key) {
  return parameters.find(key) != parameters.end();
}

