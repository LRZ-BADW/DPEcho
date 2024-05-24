//  Copyright(C) 2023 Alexander Pöppl, Intel Corp.
//  Copyright(C) 2023 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
//  with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is
//  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and limitations under the License.

#include "Logger.hpp"
#include "Output.hpp"

#ifdef MPICODE
#include <mpi.h>
#endif

#ifdef VTUNE_API_AVAILABLE
#include <ittnotify.h>
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

namespace std::filesystem {
  void create_directory(std::string name);
}

std::ofstream Log::logFile;
int Log::coutVerbosity;
int Log::clogVerbosity;
TB::Timer Log::runtimeTracker;
EndMarker const Log::endl;
FlushMarker const Log::flush;

void Log::init(std::string logfileName, int coutVerb, int clogVerb) {
  using namespace std::string_literals;
  int rank = 0;
#ifdef MPICODE
  MPI_Init(nullptr, nullptr);
  rank = mpiRank();

  if (!rank) {
    std::filesystem::create_directory(logfileName);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#else
  std::filesystem::create_directory(logfileName);
#endif
  std::string rankStr = std::to_string(rank); rankStr.insert(0, 8-rankStr.length(), '0');
  logfileName = logfileName + "/"s + rankStr;
  logFile = std::ofstream(logfileName);
  coutVerbosity = coutVerb;
  clogVerbosity = clogVerb;
  Log::Assert(logFile.good(), "Log file "s + logfileName + " failed to open."s);
  logo();
  runtimeTracker.init();
}

void Log::finalize() {
  Log::cout() << TAG << "Total runtime [s]: "<< runtimeTracker.lap() << Log::endl;
#ifdef MPICODE
  MPI_Finalize();
#endif
}


LogStream<decltype(std::cout)> const Log::cout(int verbosity) { return LogStream(!isMaster() || (verbosity > coutVerbosity), std::cout); }

LogStream<decltype(std::cerr)> const Log::cerr(int verbosity) { return LogStream(!isMaster() || (verbosity > coutVerbosity), std::cerr); }

LogStream<std::ofstream>       const Log::clog(int verbosity) {
  if (!logFile.good()) {
    throw std::logic_error("Must initialize log file before logging starts.");
  }
  return LogStream(verbosity > clogVerbosity, logFile);
}

void Log::Assert(bool condition, std::string message) {
  if (!condition) {
    Log::cerr(0) << TAG << message << Log::endl;
#ifdef MPICODE
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    std::abort();
#endif
  }
}

const std::string Log::getTag(std::string const val) {
  auto start = val.find(" ")+1, end = val.find("(");  // for funcs w type
  start = (start > end) ? 0 : start;                  // fix for funcs w/o type
  return "["+val.substr(start, end - start)+"] ";
}

int Log::mpiSize() {
  int totalRanks = 1;
#ifdef MPICODE
  MPI_Comm_size(MPI_COMM_WORLD,&totalRanks);
#endif
  return totalRanks;
};

int Log::mpiRank() {
  int rank = 0;
#ifdef MPICODE
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
  return rank;
};

int Log::mpiRanksPerNode() {
  static int cachedMpiRanksPerNode;
  if (cachedMpiRanksPerNode == 0) {
#ifdef MPICODE
    int ranksPerNode;
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    MPI_Comm_size(nodeComm, &ranksPerNode);
    cachedMpiRanksPerNode = ranksPerNode;
#else
    cachedMpiRanksPerNode = 1;
#endif

  }
  return cachedMpiRanksPerNode;
};

bool Log::isMaster() { return 0 == mpiRank(); };

void Log::barrier(){
#ifdef MPICODE
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}


void Log::togglePcontrol(int onOff){
#ifdef VTUNE_API_AVAILABLE
  if(!onOff){ __itt_pause (); }
  else      { __itt_resume(); }
#endif
#ifdef MPICODE
  MPI_Pcontrol(onOff);
#endif
}

void Log::logo(){
  Log::cout(4)<<"    ____   ____   ______       __           \n"
              <<"   / __ \\ / __ \\ / ____/_____ / /_   ____ \n"
              <<"  / / / // /_/ // __/  / ___// __ \\ / __ \\\n"
              <<" / /_/ // ____// /___ / /__ / / / // /_/ /  \n"
              <<"/_____//_/    /_____/ \\___//_/ /_/ \\____/ \n"<<Log::endl;
}


