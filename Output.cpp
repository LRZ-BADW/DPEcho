//   Copyright(C) 2021 Alexander Pöppl, Intel Corp.
//   Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include <iostream>
#include <iomanip>
#include <cstddef>
#include <cstddef>
#include <unordered_map>

#include "Output.hpp"
#include "Problem.hpp"
#include "utils/tb-types.hpp"

using namespace std;

#ifndef USE_FILE_SYSTEM_API_WORKAROUND
#include <filesystem>
#else
#include <sys/stat.h>
namespace std {
namespace filesystem {
  void create_directory(std::string name) {
    auto *log = Logger::getInstance();
    int res = mkdir(name.c_str(), 0755);
    if (res != 0 && errno != EEXIST) {
      log->Error("%s unable to create folder %s", TAG, name.c_str());
    }
  }
}
}
#endif

namespace output {

  void writeArray(std::string dir, std::string name, Problem &problem) {
    static std::unordered_map<std::string, int> numbers;
    int outNum_;
    if (numbers.find(dir + name) != numbers.end()) {
      outNum_ = numbers[dir + name];
      numbers[dir + name] += 1;
    } else {
      outNum_ = 0;
      numbers[dir + name] = 1;
      filesystem::create_directory(dir);
    }

    Logger *Log = Logger::getInstance();

    if (problem.out[0] == nullptr) {
      Log->Error("%s Array was not initialized.", TAG); return;
    }

    std::ostringstream datName, bovName;
    datName  << dir << "/" << std::setw(4)<<std::setfill('0') << outNum_;
    filesystem::create_directory(datName.str().c_str());
    bovName << datName.str() << ".bov";
    datName << "/" << name << "_" << std::setw(4)<<std::setfill('0')<< problem.BOVRank() << ".dat";
    Log->setPar(true);

    int Ncell[3], bricklets[3], Ntot=1;
    field brickSize  [3];
    field brickOrigin[3] = {problem.D_->boxMin (0), problem.D_->boxMin (1), problem.D_->boxMin (2)};

    for(int ii = 0; ii < 3; ++ii){
      if( problem.dumpHalos ){ Ncell[ii] = problem.grid_->nh[ii]; }
      else                   { Ncell[ii] = problem.grid_->n [ii]; }
      Ntot*= Ncell[ii];
      brickSize  [ii] = Ncell[ii] * problem.D_->cartDims(ii) * problem.grid_->dx[ii];
      brickOrigin[ii]*= (problem.dumpHalos) * (1.0*problem.grid_->nh[ii])/problem.grid_->n[ii];
    }

    Log->setPar(false);
    FILE *fp = fopen(datName.str().c_str(), "wb"); // For datafiles
    if(problem.out[0] != nullptr){
      for (int ii = 0; ii < Ntot; ++ii){
        for(int iVar = 0; iVar<FLD_TOT; ++iVar){
          fwrite((void *) (&(problem.out[iVar][ii])), sizeof(field), 1, fp);
        }
      }
    }
    fclose(fp);

    if( Log->master() ){ // For header file
      ofstream bov; bov.open(bovName.str(), ios_base::out);
      bov<<"TIME: "<< problem.t()<<"\n";
      bov<<"DATA_FILE: "<< std::setw(4)<<std::setfill('0') << outNum_ <<"/" << name << "_%04d.dat\n";
      bov<<"DATA_SIZE: "<< Ncell[2]*problem.D_->cartDims(2) <<" " <<Ncell[1]*problem.D_->cartDims(1)<<" "<<Ncell[0]*problem.D_->cartDims(0)<<"\n";
#ifdef SINGLE_PRECISION
      bov<<"DATA_FORMAT: FLOAT\n";
#else
      bov<<"DATA_FORMAT: DOUBLE\n";
#endif
      bov<<"VARIABLE: v\n";
      bov<<"DATA_ENDIAN: LITTLE\n";
      bov<<"CENTERING: zonal\n"; // Nicer if it was nodal, but it isn't
      bov<<"BRICK_ORIGIN: "<< brickOrigin[2] <<" "<< brickOrigin[1] <<" "<< brickOrigin[0] <<"\n";
      bov<<"BRICK_SIZE: "  << brickSize  [2] <<" "<< brickSize  [1] <<" "<< brickSize  [0] <<"\n";
      bov<<"DIVIDE_BRICK: false\n";
      bov<<"DATA_BRICKLETS: "<<Ncell     [2] <<" "<< Ncell      [1] <<" "<< Ncell      [0] <<"\n";
      bov<<"DATA_COMPONENTS: "<<FLD_TOT<<"\n"; // 1->scalar 2->complex 3->vector 4+->many scalars, *unnamed*
      bov<<std::flush; bov.close();
    }
    Log->setPar(false);  *Log+0<<TAG<<"Dumped " << dir << " (" << name << ") output #"<< outNum_++; Log->fl();
  }
}
