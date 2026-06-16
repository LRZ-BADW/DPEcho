//  Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//  Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Domain.hpp"
#include "Grid.hpp"
#include "Parameters.hpp"
#include "Logger.hpp"

using namespace sycl;

Domain::~Domain( ){
  free(bufL, qq); free(bufR, qq);
  free(sendBufL, qq); free(sendBufR, qq);
  Log::cout(4) << TAG << "Domain removed and BCex buffers deallocated." << Log::endl;
}

#define CART_DEFAULT 0

Domain::Domain(sycl::queue q, size_t bufSizes[NDIM], Parameters &param) {
  int nprocs = Log::mpiSize(), myrank = Log::mpiRank(),  reorder = 0;
  size_t bufMax = std::max({bufSizes[0], bufSizes[1], bufSizes[2]});
  qq = q; // Use the queue to allocate the buffers
  sendBufL= malloc_shared<real>(FLD_TOT*bufMax, qq); Log::Assert(sendBufL, "Cannot allocate recvBufL.");
  sendBufR= malloc_shared<real>(FLD_TOT*bufMax, qq); Log::Assert(sendBufR, "Cannot allocate recvBufR.");
  bufL= malloc_shared<real>(FLD_TOT*bufMax, qq);     Log::Assert(bufL, "Cannot allocate bufL.");
  bufR= malloc_shared<real>(FLD_TOT*bufMax, qq);     Log::Assert(bufR, "Cannot allocate bufR.");

  boxMin_[0] = param.getOr("xMin", -0.5); boxMin_[1] = param.getOr("yMin", -0.5); boxMin_[2] = param.getOr("zMin", -0.5);
  boxMax_[0] = param.getOr("xMax",  0.5); boxMax_[1] = param.getOr("yMax",  0.5); boxMax_[2] = param.getOr("zMax",  0.5);
  bcType_[0] = param.getOr("bcTypex", BCPER); bcType_[1] = param.getOr("bcTypey", BCPER); bcType_[2] = param.getOr("bcTypez", BCPER);
  cartDims_[0]=param.getOr("Rx", CART_DEFAULT); cartDims_[1] = param.getOr("Ry", CART_DEFAULT); cartDims_[2] = param.getOr("Rz", CART_DEFAULT);

  for(int iD=0; iD<NDIM; ++iD){ boxSize_[iD] = boxMax_[iD]-boxMin_[iD]; }

  // Create cartesian domain
  int cartPeriodic_[3]={1,1,1}; // MPI is broken and will never accept non-periodic, so we give it to it anyways.
  MPI_Dims_create(nprocs, 3, cartDims_);
  MPI_Cart_create(MPI_COMM_WORLD, 3, cartDims_, cartPeriodic_, reorder, &cartComm_);
  // Check that domain is consistent
  auto prod = cartDims_[0] * cartDims_[1] * cartDims_[2];
  if ( (prod) && (prod != nprocs) ){
    Log::cerr(0) << TAG << "Aborting. cartDims: " << prod << " don't match MPI ranks :" << nprocs << Log::endl;
    Log::Assert(false, "Terminating.");
  }
  MPI_Cart_coords(cartComm_, myrank, 3, cartCoords_);
  int neighCoords[3]; // One of those MPI messy interfaces. Bad code. At least it scales! - SC
  //-- Edges and neighbours
  for(unsigned short i=0; i<3; ++i){
    isEdgeLeft_ [i]=(              0 == cartCoords_[i])?1:0; // Left  edge.
    isEdgeRight_[i]=((cartDims_[i]-1)== cartCoords_[i])?1:0; // Right edge.
    for(unsigned short j=0; j<3; ++j){ neighCoords[j] = cartCoords_[j]; } // Reset
    neighCoords[i] = cartCoords_[i]-1;  MPI_Cart_rank(cartComm_, neighCoords, neighRankPrev_+i);
    neighCoords[i] = cartCoords_[i]+1;  MPI_Cart_rank(cartComm_, neighCoords, neighRankNext_+i);
  }
  //-- Initialize phisical (local!) dimensions
  for(unsigned short i=0; i<NDIM; ++i){
    locSize_[i] = boxSize_[i]/cartDims_[i]; locMin_[i] = boxMin_[i]+locSize_[i]*cartCoords_[i]; locMax_[i] = locMin_[i]+locSize_[i];
  }
  Log::clog() << TAG << "Domain created!" << Log::endl;  cartInfo();

  for(unsigned short i=0; i<NDIM; ++i){
    MPI_Send_init(sendBufL,bufSizes[i]*FLD_TOT,MPI_REAL,neighRankPrev_[i],10,cartComm_,&reqSendL[i]);
    MPI_Send_init(sendBufR,bufSizes[i]*FLD_TOT,MPI_REAL,neighRankNext_[i],20,cartComm_,&reqSendR[i]);
    MPI_Recv_init(    bufL,bufSizes[i]*FLD_TOT,MPI_REAL,neighRankNext_[i],10,cartComm_,&reqRecvL[i]);
    MPI_Recv_init(    bufR,bufSizes[i]*FLD_TOT,MPI_REAL,neighRankPrev_[i],20,cartComm_,&reqRecvR[i]);
  }

}

void Domain::cartInfo() {
  Log::cout(1) << TAG << "3D Domain: "  << cartDims_[0] << " " << cartDims_[1] << " " << cartDims_[2] << Log::endl;
  Log::clog(10) << TAG << "CartPrev: ("  << neighRankPrev_[0] << " " << neighRankPrev_[1] << " "<< neighRankPrev_[2] << ") "
                << "CartNext: (" << neighRankNext_[0] << " " << neighRankNext_[1] << " " << neighRankNext_[2] << ") " << Log::endl;
  int neighCoords_[3];
  for (int i = 0; i < 3; ++i){
    Log::clog(10) << TAG << "Prev / This / Next / Edge(l/r) [dir #" << i << "]" ;
    MPI_Cart_coords(cartComm_, neighRankPrev_[i], 3, neighCoords_);
    Log::clog(10) << "(" << neighCoords_[0] << " " << neighCoords_[1] << " " << neighCoords_[2] << ") ";
    Log::clog(10) << "(" << cartCoords_[0]  << " " << cartCoords_[1]  << " " << cartCoords_[2]  << ") ";
    MPI_Cart_coords(cartComm_, neighRankNext_[i], 3, neighCoords_);
    Log::clog(10) << "(" << neighCoords_[0] << " " << neighCoords_[1] << " " << neighCoords_[2] <<") ";
    Log::clog(10) << "(" << isEdgeLeft_[i]  << "/" << isEdgeRight_[i] <<") ";
    Log::clog(10) << Log::endl;
  }
}

void Domain::boxInfo() {
  Log::cout(1) << TAG << "Box  size (" << boxSize_[0] << " " << boxSize_[1] << " " << boxSize_[2]
               << ") From ("   << boxMin_ [0] << " " << boxMin_ [1] << " " << boxMin_ [2]
               << ") To ("     << boxMax_ [0] << " " << boxMax_ [1] << " " << boxMax_ [2] << ")" << Log::endl;
}

void Domain::locInfo() {
  Log::clog(1) <<TAG << "Local size (" << locSize_[0] << " " << locSize_[1] << " " << locSize_[2]
               << ") From ("   << locMin_ [0] << " " << locMin_ [1] << " " << locMin_ [2]
               << ") To ("     << locMax_ [0] << " " << locMax_ [1] << " " << locMax_ [2] << ")" << Log::endl;
}

// Variables are passed, so any array can be used.
// ACHTUNG:
//  - It always assumes periodic, w or w/o MPI. At the end it will take care of other BCs.
void Domain::BCex(int myDir, Grid gr, real_array &v, int dType){ // gr is the usual local grid.

  MPI_Status  status;
  int i0 = (dType==BCEX_VU)?0:1; // Flux is Mx+1 so bcex needs shift

  // Each direction may have a different buffer size. We may also store them, whatever.
  int   nBuf[]=      {gr.nh[0], gr.nh[1], gr.nh[2]}; nBuf[myDir] = gr.h[myDir];
  int   nOff[]=      {       0,        0,        0}; nOff[myDir] = gr.h[myDir];
  range rBuf  = range( nBuf[0],  nBuf[1],  nBuf[2]); auto   sBuf = rBuf.size(); auto rPlane=rBuf; // rPlane is for BCOF3
  nd_range<3> ndr = getMatchingNdRange(rBuf, range<3>(4,4,4));
  Log::clog(8) << TAG << " Filling buffers ..." << Log::endl;

  real *bL = this->sendBufL, *bR = this->sendBufR;
  MPI_Start(&reqRecvL[myDir]);
  MPI_Start(&reqRecvR[myDir]);
  id<3> nOffRead = range<3>(nOff[0], nOff[1], nOff[2]); nOffRead[myDir] +=i0;
  range<3> const fullGrR(gr.nh[0], gr.nh[1], gr.nh[2]); 
  qq.parallel_for( ndr, [=](nd_item<3> it){
    id<3> id = it.get_global_id();
    if (isOutOfBounds(id, rBuf)) return;
    size_t iBufL = globLinId(id, rBuf, sycl::id<3>(0,0,0)), iBufR = sBuf   -1 -iBufL; // The same, if we start from the end
    size_t iVL   = globLinId(id, fullGrR, nOffRead), iVR   = gr.nht -1 -iVL;
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      bL[iVar*sBuf+iBufL] = v[iVar][iVL];
    }
  }).wait_and_throw();
  MPI_Start(&reqSendL[myDir]);
  qq.parallel_for( rBuf, [=](item<3> it){
    int iBufL = it.get_linear_id()        , iBufR = sBuf   -1 -iBufL;
    int iVL   = globLinId(it, fullGrR, nOffRead), iVR   = gr.nht -1 -iVL;
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      bR[iVar*sBuf+iBufR] = v[iVar][iVR];
    }
  }).wait_and_throw();

//- Middle communication
  MPI_Start(&reqSendR[myDir]);
  MPI_Wait (&reqRecvL[myDir],&status);
  Log::clog(8) << TAG << " Recopying from buffers..." << Log::endl;
  nOff[myDir] = 0; // To write, we start from 0
  id<3> nOffW = range<3>(nOff[0], nOff[1], nOff[2]);
  qq.parallel_for(ndr,[=,bL=this->bufL](nd_item<3> it){  // v -> WHindex
    id<3> id = it.get_global_id();
    if (isOutOfBounds(id, rBuf)) return;
    size_t iVL   = globLinId(id, fullGrR, nOffW), iVR   = gr.nht -1 -iVL  ;  // For the regular BCEX
    size_t iBufL = globLinId(id, rBuf, sycl::id<3>(0,0,0)), iBufR = sBuf   -1 -iBufL;  // The same, if we start from the end
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      v[iVar][iVR] = bL[iVar*sBuf+iBufR];  // ...besides the flipped assignments
    }
  }); // NO SYCL wait here!
  MPI_Wait (&reqRecvR[myDir],&status);
  qq.parallel_for(rBuf,[=,bR=this->bufR](item<3> it){
    int iVL   = globLinId(it, fullGrR, nOffW), iVR   = gr.nht -1 -iVL;
    int iBufL = it.get_linear_id()        , iBufR = sBuf   -1 -iBufL;
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      v[iVar][iVL] = bR[iVar*sBuf+iBufL];
    }
  }).wait_and_throw();
  switch(bcType_[myDir]){ //-- PROCESSING BC TYPEs
    case BCOF0: //- Outflow w. 0th order interp
      Log::clog(10) << TAG << " Processing Outflow BCs of order 0..." << Log::endl;
      if(isEdgeLeft_[myDir]){
        qq.parallel_for(rBuf,[=](item<3> it){  // v -> WHindex
          id<3> readId, writeId = readId = it.get_id()  ;  readId[myDir]+= gr.h[myDir] - it.get_id(myDir);
          int iVL = globLinId(writeId, gr.nh, nOff)     ,           iOut = globLinId(readId, gr.nh, nOff);
          for(int iVar=0; iVar<FLD_TOT; ++iVar){ v[iVar][iVL] = v[iVar][iOut]; };
        }).wait_and_throw();
      }
      if(isEdgeRight_[myDir]){
        qq.parallel_for(rBuf,[=](item<3> it){  // v -> WHindex
          id<3> gridOffset = id(0,0,0)                             ; gridOffset[myDir] =   gr.n[myDir] + gr.h[myDir];
          id<3> readId, writeId = readId = it.get_id() + gridOffset;     readId[myDir] = readId[myDir] - it.get_id(myDir) - 1;
          int       iVR = globLinId(writeId, gr.nh, nOff)  ,                      iOut = globLinId(readId, gr.nh, nOff);
          for(int iVar=0; iVar<FLD_TOT; ++iVar){ v[iVar][iVR] = v[iVar][iOut]; };
        }).wait_and_throw();
      }
      break;
      // ACHTUNG:: MOST LIKELY INCORRECT!!!!
    case BCOF3: //- Outflow w. 3rd order interp
      Log::clog(10) << TAG << " Processing Outflow BCs of order 3..." << Log::endl;
      if(isEdgeLeft_[myDir]){
        for(int iVar=0; iVar<FLD_TOT; ++iVar){
          qq.parallel_for(rPlane,[=](item<3> it){  // v -> WHindex
            id<3> myId = it.get_id();
            int iVL = globLinId(myId, gr.nh, nOff), step = stride(myId, myDir, gr.nh);
            for(int iLay=0; iLay<gr.h[myDir]; ++iLay){
               v[iVar][iVL] = -1*v[iVar][iVL+step] -3*v[iVar][iVL+2*step] + v[iVar][iVL+3*step];
               iVL+=-step;
            }
          });
        }; qq.wait_and_throw();
      }
      if(isEdgeRight_[myDir]){
        qq.parallel_for(rBuf,[=](item<3> it){  // v -> WHindex
          id<3> gridOffset = id(0,0,0);  gridOffset[myDir] = gr.n[myDir] + gr.h[myDir];
          id<3> myId = it.get_id() + gridOffset;
          int iVL = globLinId(myId, gr.nh, nOff), step = stride(myId, myDir, gr.nh);
          for(int iVar=0; iVar<FLD_TOT; ++iVar){
          for(int iLay=0; iLay<gr.h[myDir]; ++iLay){
             v[iVar][iVL] = -1*v[iVar][iVL-step] -3*v[iVar][iVL-2*step] + v[iVar][iVL-3*step];
             iVL+= step;
          }}
        }).wait_and_throw();
      }
      break;
    case BCPER:  break; //- Periodic; do nothing
    default   : Log::cerr(2) << TAG << "Unknown BC TYPE " << bcType_[myDir] << " along direction " << myDir << ". Proceeding as periodic." << Log::endl;
  }// End switch

  MPI_Wait(&reqSendL[myDir],&status);
  MPI_Wait(&reqSendR[myDir],&status);
  Log::clog(8) << TAG << " BCex complete." << Log::endl;
  return;
}
