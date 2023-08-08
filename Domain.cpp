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


#if SYCL == OpenSYCL && SYCL_ARCH == AMD
// TODO: This should also trigger on if GPU aware MPI is enabled
// For MPI enabled and disabled, performance is enhanced if the buffer is located on device
#define USE_DEVICE_BUFFER 1
#ifdef MPICODE
#define USE_ISENDRECV 1
#endif
#endif


Domain::~Domain( ){  free(bufL, qq); free(bufR, qq); free(rbufR, qq); free(rbufR, qq);
  Log->setPar(false); *Log+4<<TAG<<"Domain removed and BCex buffers deallocated."; Log->fl();
}

Domain::Domain(mysycl::queue q, size_t bufMax) {
  Log = Logger::getInstance();  Log->setPar(false);
  int nprocs  = Log->getNumProcs(), myrank = Log->getMyRank(),  reorder = 0;

  qq = q; // Use the queue to allocate the buffers
#ifdef USE_DEVICE_BUFFER
  bufL= malloc_device<field>(FLD_TOT*bufMax, qq); if(!bufL){Log->Error("%s Cannot allocate bufL.", TAG );}
  bufR= malloc_device<field>(FLD_TOT*bufMax, qq); if(!bufR){Log->Error("%s Cannot allocate bufR.", TAG );}
#else
  bufL= malloc_shared<field>(FLD_TOT*bufMax, qq); if(!bufL){Log->Error("%s Cannot allocate bufL.", TAG );}
  bufR= malloc_shared<field>(FLD_TOT*bufMax, qq); if(!bufR){Log->Error("%s Cannot allocate bufR.", TAG );}
#endif

#ifdef USE_ISENDRECV
  rbufL= malloc_device<field>(FLD_TOT*bufMax, qq); if(!rbufL){Log->Error("%s Cannot allocate rbufL.", TAG );}
  rbufR= malloc_device<field>(FLD_TOT*bufMax, qq); if(!rbufR){Log->Error("%s Cannot allocate rbufR.", TAG );}
#else
  rbufL = rbufR = nullptr;
#endif

  // Assign default values before reading parameters
  boxMin_[0] = boxMin_[1] = boxMin_[2] =-0.5;
  boxMax_[0] = boxMax_[1] = boxMax_[2] = 0.5;
  bcType_[0] = bcType_[1] = bcType_[2] = BCPER;
#ifdef MPICODE
  cartDims_[0] = cartDims_[1] = cartDims_[2] = 0;
#else
  cartDims_[0] = cartDims_[1] = cartDims_[2] = 1;
#endif
  // Now read the parameters
  std::string key, value;   std::ifstream inFile("echo.par");  *Log+0<<TAG<<" Reading input: ";
  while (std::getline(inFile, key, ' ') && std::getline(inFile, value) ){
    if(!key.compare("xMin"   )){ boxMin_[0] = std::stof(value); *Log<<"\n\txMin    "<<boxMin_[0] <<" "; continue;}
    if(!key.compare("yMin"   )){ boxMin_[1] = std::stof(value); *Log<<"\n\tyMin    "<<boxMin_[1] <<" "; continue;}
    if(!key.compare("zMin"   )){ boxMin_[2] = std::stof(value); *Log<<"\n\tzMin    "<<boxMin_[2] <<" "; continue;}
    if(!key.compare("xMax"   )){ boxMax_[0] = std::stof(value); *Log<<"\n\txMax    "<<boxMax_[0] <<" "; continue;}
    if(!key.compare("yMax"   )){ boxMax_[1] = std::stof(value); *Log<<"\n\tyMax    "<<boxMax_[1] <<" "; continue;}
    if(!key.compare("zMax"   )){ boxMax_[2] = std::stof(value); *Log<<"\n\tzMax    "<<boxMax_[2] <<" "; continue;}
    if(!key.compare("bcTypex")){ bcType_[0] = std::stoi(value); *Log<<"\n\tbcTypex "<<bcType_[0] <<" "; continue;}
    if(!key.compare("bcTypey")){ bcType_[1] = std::stoi(value); *Log<<"\n\tbcTypey "<<bcType_[1] <<" "; continue;}
    if(!key.compare("bcTypez")){ bcType_[2] = std::stoi(value); *Log<<"\n\tbcTypez "<<bcType_[2] <<" "; continue;}
#ifdef MPICODE
    if(!key.compare("Rx")){ cartDims_[0] = std::stoi(value); *Log<<"\n\tcartDimx "<<cartDims_[0]<<" "; continue;}
    if(!key.compare("Ry")){ cartDims_[1] = std::stoi(value); *Log<<"\n\tcartDimy "<<cartDims_[1]<<" "; continue;}
    if(!key.compare("Rz")){ cartDims_[2] = std::stoi(value); *Log<<"\n\tcartDimz "<<cartDims_[2]<<" "; continue;}
#else
    cartCoords_[0] = cartCoords_[1] = cartCoords_[2] = 0;
#endif
  }; Log->fl();
  boxSize_[0] = boxMax_[0]-boxMin_[0];  boxSize_[1] = boxMax_[1]-boxMin_[1];  boxSize_[2] = boxMax_[2]-boxMin_[2];
  cartPeriodic_[0] = cartPeriodic_[1] = cartPeriodic_[2] = 1; // TODO Always periodic for now

#ifdef MPICODE
  // Create cartesian domain
  MPI_Dims_create(nprocs, 3, cartDims_);
  MPI_Cart_create(MPI_COMM_WORLD, 3, cartDims_, cartPeriodic_, reorder, &cartComm_);
  // Check that domain is consistent
  unsigned const prod = cartDims_[0] * cartDims_[1] * cartDims_[2];
  if ( (prod) && (prod != nprocs) ){
    *Log+0<<TAG<<"Aborting. cartDims: "<<prod<<" don't match MPI ranks :"<<nprocs;
    Log->fl(); Log->Error("");
  }
  MPI_Cart_coords(cartComm_, myrank, 3, cartCoords_);
  int neighCoords[3]; // One of those MPI messy interfaces. Bad code. At least it scales! - SC
  neighCoords[1] = cartCoords_[1]; neighCoords[2] = cartCoords_[2];  // X-neighbors
  neighCoords[0] = cartCoords_[0]-1; MPI_Cart_rank(cartComm_, neighCoords, neighRankPrev_+0);
  neighCoords[0] = cartCoords_[0]+1; MPI_Cart_rank(cartComm_, neighCoords, neighRankNext_+0);

  neighCoords[0] = cartCoords_[0]; neighCoords[2] = cartCoords_[2];  // Y-neighbors
  neighCoords[1] = cartCoords_[1]-1; MPI_Cart_rank(cartComm_, neighCoords, neighRankPrev_+1);
  neighCoords[1] = cartCoords_[1]+1; MPI_Cart_rank(cartComm_, neighCoords, neighRankNext_+1);

  neighCoords[0] = cartCoords_[0]; neighCoords[1] = cartCoords_[1];  // Z-neighbors
  neighCoords[2] = cartCoords_[2]-1; MPI_Cart_rank(cartComm_, neighCoords, neighRankPrev_+2);
  neighCoords[2] = cartCoords_[2]+1; MPI_Cart_rank(cartComm_, neighCoords, neighRankNext_+2);
#endif

  // Initialize phisical (local!) dimensions
  locSize_[0] = boxSize_[0]/cartDims_[0]; locMin_[0] = boxMin_[0]+locSize_[0]*cartCoords_[0]; locMax_[0] = locMin_[0]+locSize_[0];
  locSize_[1] = boxSize_[1]/cartDims_[1]; locMin_[1] = boxMin_[1]+locSize_[1]*cartCoords_[1]; locMax_[1] = locMin_[1]+locSize_[1];
  locSize_[2] = boxSize_[2]/cartDims_[2]; locMin_[2] = boxMin_[2]+locSize_[2]*cartCoords_[2]; locMax_[2] = locMin_[2]+locSize_[2];

  *Log+0<<TAG<<"Domain created!"; Log->fl();  cartInfo();
}

void Domain::cartInfo() {
  Log->setPar(false);
  *Log+1<<TAG<<"3D Domain: " <<cartDims_[0]<<" "<<cartDims_[1]<<" "<<cartDims_[2]; Log->fl();
  Log->setPar(true);

  *Log+1<<TAG <<"CartCoords:    (" <<cartCoords_   [0] <<" "<< cartCoords_   [1] <<" "<< cartCoords_   [2] <<") "
              <<"CartPeriodic:  (" <<cartPeriodic  (0) <<" "<< cartPeriodic  (1) <<" "<< cartPeriodic  (2) <<")";  Log->fl();
#ifdef MPICODE
  *Log+10<<TAG<<"CartPrev: (" <<neighRankPrev_[0] <<" "<< neighRankPrev_[1] <<" "<< neighRankPrev_[2] <<") "
              <<"CartNext: (" <<neighRankNext_[0] <<" "<< neighRankNext_[1] <<" "<< neighRankNext_[2] <<") "; Log->fl();
  int neighCoords_[3];
  for (int i = 0; i < 3; ++i){
    *Log+10<<TAG <<"Prev / This / Next ["<<i<<"]" ;
    MPI_Cart_coords(cartComm_, neighRankPrev_[i], 3, neighCoords_);
    *Log<<"("<< neighCoords_   [0] <<" "<< neighCoords_   [1] <<" "<< neighCoords_   [2] <<") ";
    *Log<<"("<< cartCoords_    [0] <<" "<< cartCoords_    [1] <<" "<< cartCoords_    [2] <<") ";
    MPI_Cart_coords(cartComm_, neighRankNext_[i], 3, neighCoords_);
    *Log<<"("<< neighCoords_   [0] <<" "<< neighCoords_   [1] <<" "<< neighCoords_   [2] <<") ";
    Log->fl();
  }
#endif
}

void Domain::boxInfo() {
  Log->setPar(false);
  *Log+1<<TAG<<"Box  size ("<<boxSize_[0]<<" "<<boxSize_[1]<<" "<<boxSize_[2]
             <<") From ("   <<boxMin_ [0]<<" "<<boxMin_ [1]<<" "<<boxMin_ [2]
             <<") To ("     <<boxMax_ [0]<<" "<<boxMax_ [1]<<" "<<boxMax_ [2]<<")";  Log->fl();
}

void Domain::locInfo() {
  Log->setPar(true);
  *Log+1<<TAG<<"Local size ("<<locSize_[0]<<" "<<locSize_[1]<<" "<<locSize_[2]
             <<") From ("    <<locMin_ [0]<<" "<<locMin_ [1]<<" "<<locMin_ [2]
             <<") To ("      <<locMax_ [0]<<" "<<locMax_ [1]<<" "<<locMax_ [2]<<")";  Log->fl();
}

// Variables are passed, so any array can be used.
// ACHTUNG:
//  - It does fields one by one; must call it FLD_TOT times, or use the wrapper "fillAllHalos"
//  - It assumes periodic, w or w/o MPI. For clarity;
//    - Periodic?            Call me and you're done.
//    - Non-periodic MPI?    Call me + appropriate BC fill.
//    - Non-periodic serial? Don't call me, only BC fill.
void Domain::BCex(int myDir, Grid gr, field_array &v){ // gr is the usual local grid.

#ifdef MPICODE
  int myRank; MPI_Comm_rank(cartComm_, &myRank);
  MPI_Status status;
  
#define CHECK_MPI(status) \
  {int mympistatus = status; if(mympistatus != MPI_SUCCESS) {std::cerr << "*** ERROR *** " << __FILE__ << ":"<<__LINE__<<"\n\n" << #status << "\n\nReturned status: " << mympistatus << "\n"; MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);}}
#endif

  // Each direction may have a different buffer size. We may also store them, whatever.
  int   nBuf[]=      {gr.nh[0], gr.nh[1], gr.nh[2]}; nBuf[myDir] = gr.h[myDir];
  int   nOff[]=      {       0,        0,        0}; nOff[myDir] = gr.h[myDir]; // nOff = gr.h left corners out
  range rBuf  = range( nBuf[0],  nBuf[1],  nBuf[2]); auto   sBuf = rBuf.size();

  Log->setPar(true); *Log+8<<TAG<<" Filling buffers ..."; Log->fl();
  
#ifdef MPICODE // In serial, it's wasteful to allocate the buffers. As if someone cared about serial cases. -SC
#ifdef USE_ISENDRECV
  // (Aug2023) MPI appears to require a full queue sync - otherwise we see memory corruption issues
  qq.wait();

  *Log+8<<TAG<<"L+R MPI_Irecv ... "; Log->fl();
  auto lower_recv_request = MPI_Request();
  CHECK_MPI(MPI_Irecv(this->rbufL,FLD_TOT*sBuf,MPI_FIELD,neighRankPrev_[myDir],0,cartComm_,&lower_recv_request));
  auto upper_recv_request = MPI_Request();
  CHECK_MPI(MPI_Irecv(this->rbufR,FLD_TOT*sBuf,MPI_FIELD,neighRankNext_[myDir],0,cartComm_,&upper_recv_request));
#endif
#endif

  // On the parallel_for: capture the buffers separately.
  // Also: cumbersome to calculate rLoc (quite small anyway); with item<3> runtime decides.
  qq.parallel_for<class parForPack>( rBuf, [=, send_bufL=this->bufL, send_bufR=this->bufR ](item<3> it){
    int iBufL = it.get_linear_id()        , iBufR = sBuf   -1 -iBufL; // The same, if we start from the end
    int iVL   = globLinId(it, gr.nh, nOff), iVR   = gr.nht -1 -iVL  ;
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      send_bufL[iVar*sBuf+iBufL] = v[iVar][iVL];
      send_bufR[iVar*sBuf+iBufR] = v[iVar][iVR];
    }
  }).wait_and_throw();

  // Reverse the buffers for the unpacking phase
  field * recvBufL = this->bufR;
  field * recvBufR = this->bufL;

#ifdef MPICODE
#ifdef USE_ISENDRECV
  // (Aug2023) MPI appears to require a full queue sync - otherwise we see memory corruption issues
  qq.wait();

  *Log+8<<TAG<<"L+R MPI_Isend ... "; Log->fl();
  auto lower_send_request = MPI_Request();
  CHECK_MPI(MPI_Isend(this->bufL,FLD_TOT*sBuf,MPI_FIELD,neighRankPrev_[myDir],0,cartComm_,&lower_send_request));
  auto upper_send_request = MPI_Request();
  CHECK_MPI(MPI_Isend(this->bufR,FLD_TOT*sBuf,MPI_FIELD,neighRankNext_[myDir],0,cartComm_,&upper_send_request));

  CHECK_MPI(MPI_Wait(&lower_recv_request,&status));
  CHECK_MPI(MPI_Wait(&upper_recv_request,&status));
 
  recvBufL = this->rbufR;
  recvBufR = this->rbufL;
#else

  *Log+8<<TAG<<"L+R MPI_Sendrrecv_replace ... "; Log->fl();
  CHECK_MPI(MPI_Sendrecv_replace(bufL,sBuf*FLD_TOT,MPI_FIELD,neighRankPrev_[myDir],myRank,neighRankNext_[myDir],MPI_ANY_TAG,cartComm_,&status));
  CHECK_MPI(MPI_Sendrecv_replace(bufR,sBuf*FLD_TOT,MPI_FIELD,neighRankNext_[myDir],myRank,neighRankPrev_[myDir],MPI_ANY_TAG,cartComm_,&status));

#endif

#endif

  nOff[myDir] = 0; // To write, we start from 0
  Log->setPar(true); *Log+8<<TAG<<" Recopying from buffers..."; Log->fl();
  qq.parallel_for<class parForUnpack>( rBuf, [=](item<3> it){  // v -> WHindex
    int iBufL = it.get_linear_id(        ), iBufR = sBuf   -1 -iBufL;  // The same, if we start from the end
    int iVL   = globLinId(it, gr.nh, nOff), iVR   = gr.nht -1 -iVL  ;
    for(int iVar=0; iVar<FLD_TOT; ++iVar){
      v[iVar][iVL] = recvBufL[iVar*sBuf+iBufL];
      v[iVar][iVR] = recvBufR[iVar*sBuf+iBufR];
    }
  }).wait_and_throw();

#ifdef MPICODE
#ifdef USE_ISENDRECV
  // (Aug2023) MPI appears to require a full queue sync - otherwise we see memory corruption issues
  qq.wait();
  CHECK_MPI(MPI_Wait(&lower_send_request,&status));
  CHECK_MPI(MPI_Wait(&upper_send_request,&status));
#endif
#endif

}
