//  Copyright(C) 2023 Alexander Pöppl, Intel Corp.
//  Copyright(C) 2023 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
//  with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is
//  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include "utils/tb-timer.hpp"

#include <fstream>
#include <iostream>

#define TAG Log::getTag(__PRETTY_FUNCTION__)
#define KTAG __FILE__ // Usable inside DPC++ kernels. For lack of a better one

class EndMarker {};
class FlushMarker {};

template <typename UnderlyingStream>
class LogStream {
  friend class Log;

  bool isSilent;
  UnderlyingStream &logFile;

  LogStream(bool isSilent, UnderlyingStream &logFile)
    : isSilent(isSilent),
      logFile(logFile) {};

  public:
    template <typename T> LogStream const operator<<(T val) const {
      if (!isSilent) { logFile << val; }
      return *this;
    }

    LogStream const operator<<(FlushMarker) const {
      if (!isSilent) {
        logFile << std::flush;
      }
      return *this;
    }

    LogStream const operator<<(EndMarker) const {
      if (!isSilent) {
        logFile << std::endl;
      }
      return *this;
    }
};


class Log {
  static std::ofstream logFile;
  static int coutVerbosity;
  static int clogVerbosity;
  static TB::Timer runtimeTracker;

  public:

  static void init(std::string logfileName, int coutVerb, int clogVerb);
  static void finalize();
  static void logo();

  static EndMarker const endl;
  static FlushMarker const flush;

  static LogStream<decltype(std::cout)> const cout(int verbosity=0);
  static LogStream<decltype(std::cerr)> const cerr(int verbosity=0);
  static LogStream<std::ofstream>       const clog(int verbosity=0);
  static void Assert(bool condition, std::string message);
  static const std::string getTag(std::string const val);
  static void togglePcontrol(int onOff);

  // MPI Helpers (avoid some ifdefs)
  static int mpiSize();
  static int mpiRanksPerNode();
  static int mpiRank();
  static bool isMaster();
  static void barrier();
};
