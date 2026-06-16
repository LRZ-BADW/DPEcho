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

#include "Parameters.hpp"
#include "echo.hpp"
#include "Grid.hpp"
#include "utils/tb-types.hpp"

#include <sycl/sycl.hpp>
#include <mpi.h>

#define BCEX_VU 0
#define BCEX_FL 1

class Domain {
  private:
    sycl::queue qq;
    int cartDims_[NDIM], cartCoords_[NDIM], bcType_[NDIM];
    bool isEdgeLeft_[NDIM], isEdgeRight_[NDIM];
    real boxMin_[NDIM], boxMax_[NDIM], boxSize_[NDIM]; // Global info, physical
    real locMin_[NDIM], locMax_[NDIM], locSize_[NDIM]; // This rank info, physical
    real *bufL, *bufR;                                 // Buffers for boundary XCHG
    int neighRankPrev_[NDIM], neighRankNext_[NDIM];  MPI_Comm cartComm_;
    real *sendBufL, *sendBufR;                // Additional buffers, as sendRecv needs two
    MPI_Request reqSendL[NDIM], reqSendR[NDIM], reqRecvL[NDIM], reqRecvR[NDIM];

  public:
    Domain(sycl::queue, size_t[NDIM], Parameters &);
    ~Domain( );
    void BCex (int direction, Grid gr, real_array &v, int dType=BCEX_VU); // gr is the usual local grid.
    void cartInfo();
    void boxInfo();
    void locInfo();
    inline bool  isEdgeLeft (unsigned i){return isEdgeLeft_  [i]; };
    inline bool  isEdgeRight(unsigned i){return isEdgeRight_ [i]; };
    inline int   cartCoords (unsigned i){return cartCoords_  [i]; };
    inline int   cartDims   (unsigned i){return cartDims_    [i]; };
    inline real boxMin     (unsigned i){return boxMin_      [i]; };
    inline real boxMax     (unsigned i){return boxMax_      [i]; };
    inline real boxSize    (unsigned i){return boxSize_     [i]; };
    inline real locMin     (unsigned i){return locMin_      [i]; };
    inline real locMax     (unsigned i){return locMax_      [i]; };
    inline real locSize    (unsigned i){return locSize_     [i]; };
    inline MPI_Comm cartComm(          ){return cartComm_       ; }; // Just in case

};
#endif
