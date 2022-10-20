//   Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//   Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Domain_hpp_
#define _Domain_hpp_

#include "echo.hpp"
#include "Logger.hpp"
#include "Grid.hpp"
#include "utils/tb-types.hpp"
#include "echoSycl.hpp"

class Domain {
  private:
    Logger *Log;
    mysycl::queue qq;  field *bufL, *bufR; // Tools for the BCex
    // Most of the following could be public const and save us the functions below
    // IF we pass values from constructor instead of reading echo.par in there.
    int cartDims_[3], cartPeriodic_[3], cartCoords_[3], bcType_[3];
    field boxMin_[3], boxMax_[3], boxSize_[3]; // Global info, physical
    field locMin_[3], locMax_[3], locSize_[3]; // This rank info, physical
#ifdef MPICODE
    int neighRankPrev_[3], neighRankNext_[3];  MPI_Comm cartComm_;
#endif

  public:
    Domain(mysycl::queue, size_t);
    ~Domain( );
    void cartInfo();
    inline int cartCoords  (unsigned i){return cartCoords_  [i]; };
    inline int cartDims    (unsigned i){return cartDims_    [i]; };
    inline int cartPeriodic(unsigned i){return cartPeriodic_[i]; };
    void boxInfo();
    inline field boxMin    (unsigned i){return boxMin_      [i]; };
    inline field boxMax    (unsigned i){return boxMax_      [i]; };
    inline field boxSize   (unsigned i){return boxSize_     [i]; };
    void  locInfo();
    inline field locMin    (unsigned i){return locMin_      [i]; };
    inline field locMax    (unsigned i){return locMax_      [i]; };
    inline field locSize   (unsigned i){return locSize_     [i]; };
    void  BCex (int direction, Grid gr, field_array &v); // gr is the usual local grid.

#ifdef MPICODE
    inline MPI_Comm cartComm(         ){return cartComm_       ; }; // Just in case
#endif

};
#endif
