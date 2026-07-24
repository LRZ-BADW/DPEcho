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

#include <mdspan.hpp>

using namespace sycl;

Domain::~Domain( ){
  free(bufL, qq); free(bufR, qq);
  free(sendBufL, qq); free(sendBufR, qq);
  Log::cout(4) << TAG << "Domain removed and BCex buffers deallocated." << Log::endl;
}

#define CART_DEFAULT 0

Domain::Domain(sycl::queue q, size_t bufSizes[NDIM], Parameters &param, int nFields) {
  int nprocs = Log::mpiSize(), myrank = Log::mpiRank(),  reorder = 0;
  size_t bufMax = std::max({bufSizes[0], bufSizes[1], bufSizes[2]});
  nFields_ = nFields;
  qq = q; // Use the queue to allocate the buffers
  sendBufL= malloc_shared<real>(nFields_*bufMax, qq); Log::Assert(sendBufL, "Cannot allocate recvBufL.");
  sendBufR= malloc_shared<real>(nFields_*bufMax, qq); Log::Assert(sendBufR, "Cannot allocate recvBufR.");
  bufL= malloc_shared<real>(nFields_*bufMax, qq);     Log::Assert(bufL, "Cannot allocate bufL.");
  bufR= malloc_shared<real>(nFields_*bufMax, qq);     Log::Assert(bufR, "Cannot allocate bufR.");

  boxMin_[0] = param.getOr("xMin", -0.5); boxMin_[1] = param.getOr("yMin", -0.5); boxMin_[2] = param.getOr("zMin", -0.5);
  boxMax_[0] = param.getOr("xMax",  0.5); boxMax_[1] = param.getOr("yMax",  0.5); boxMax_[2] = param.getOr("zMax",  0.5);
  bcType_[0] = param.getOr("bcTypex", BCPER); bcType_[1] = param.getOr("bcTypey", BCPER); bcType_[2] = param.getOr("bcTypez", BCPER);
  cartDims_[0]=param.getOr("Rx", CART_DEFAULT); cartDims_[1] = param.getOr("Ry", CART_DEFAULT); cartDims_[2] = param.getOr("Rz", CART_DEFAULT);

  for(int iD=0; iD<NDIM; ++iD){ boxSize_[iD] = boxMax_[iD]-boxMin_[iD]; }

  // Create cartesian domain
  int cartPeriodic_[NDIM]={1,1,1}; // MPI is broken and will never accept non-periodic, so we give it to it anyways.
  MPI_Dims_create(nprocs, NDIM, cartDims_);
  MPI_Cart_create(MPI_COMM_WORLD, NDIM, cartDims_, cartPeriodic_, reorder, &cartComm_);
  // Check that domain is consistent
  auto prod = cartDims_[0] * cartDims_[1] * cartDims_[2];
  if ( (prod) && (prod != nprocs) ){
    Log::cerr(0) << TAG << "Aborting. cartDims: " << prod << " don't match MPI ranks :" << nprocs << Log::endl;
    Log::Assert(false, "Terminating.");
  }
  MPI_Cart_coords(cartComm_, myrank, NDIM, cartCoords_);
  int neighCoords[NDIM]; // One of those MPI messy interfaces. Bad code. At least it scales! - SC
  //-- Edges and neighbours
  for(unsigned short i=0; i<NDIM; ++i){
    isEdgeLeft_ [i]=(              0 == cartCoords_[i])?1:0; // Left  edge.
    isEdgeRight_[i]=((cartDims_[i]-1)== cartCoords_[i])?1:0; // Right edge.
    for(unsigned short j=0; j<NDIM; ++j){ neighCoords[j] = cartCoords_[j]; } // Reset
    neighCoords[i] = cartCoords_[i]-1;  MPI_Cart_rank(cartComm_, neighCoords, neighRankPrev_+i);
    neighCoords[i] = cartCoords_[i]+1;  MPI_Cart_rank(cartComm_, neighCoords, neighRankNext_+i);
  }
  //-- Initialize physical (local!) dimensions
  for(unsigned short i=0; i<NDIM; ++i){
    locSize_[i] = boxSize_[i]/cartDims_[i]; locMin_[i] = boxMin_[i]+locSize_[i]*cartCoords_[i]; locMax_[i] = locMin_[i]+locSize_[i];
  }
  Log::clog() << TAG << "Domain created!" << Log::endl;  cartInfo();

  for(unsigned short i=0; i<NDIM; ++i){
    MPI_Send_init(sendBufL,bufSizes[i]*nFields_,MPI_REAL,neighRankPrev_[i],10,cartComm_,&reqSendL[i]);
    MPI_Send_init(sendBufR,bufSizes[i]*nFields_,MPI_REAL,neighRankNext_[i],20,cartComm_,&reqSendR[i]);
    MPI_Recv_init(    bufL,bufSizes[i]*nFields_,MPI_REAL,neighRankNext_[i],10,cartComm_,&reqRecvL[i]);
    MPI_Recv_init(    bufR,bufSizes[i]*nFields_,MPI_REAL,neighRankPrev_[i],20,cartComm_,&reqRecvR[i]);
  }

}

void Domain::cartInfo() {
  Log::cout(1) << TAG << "3D Domain: "  << cartDims_[0] << " " << cartDims_[1] << " " << cartDims_[2] << Log::endl;
  Log::clog(10) << TAG << "CartPrev: ("  << neighRankPrev_[0] << " " << neighRankPrev_[1] << " "<< neighRankPrev_[2] << ") "
                << "CartNext: (" << neighRankNext_[0] << " " << neighRankNext_[1] << " " << neighRankNext_[2] << ") " << Log::endl;
  int neighCoords_[NDIM];
  for (int i = 0; i < NDIM; ++i){
    Log::clog(10) << TAG << "Prev / This / Next / Edge(l/r) [dir #" << i << "]" ;
    MPI_Cart_coords(cartComm_, neighRankPrev_[i], NDIM, neighCoords_);
    Log::clog(10) << "(" << neighCoords_[0] << " " << neighCoords_[1] << " " << neighCoords_[2] << ") ";
    Log::clog(10) << "(" << cartCoords_[0]  << " " << cartCoords_[1]  << " " << cartCoords_[2]  << ") ";
    MPI_Cart_coords(cartComm_, neighRankNext_[i], NDIM, neighCoords_);
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
  // WARNING:
//  - It always assumes periodic, w or w/o MPI. At the end it will take care of other BCs.
void Domain::BCex(int myDir, Grid gr, real_array &v, int dType){ // gr is the usual local grid.

  namespace ms = std::experimental;
  using dex3 = ms::dextents<size_t, 3>;
  using dex4 = ms::dextents<size_t, 4>;

  MPI_Status  status;
  int i0 = (dType==BCEX_VU)?0:1; // Flux is Mx+1 so bcex needs shift
  int nFields = this->nFields_; // local copy for SYCL kernel capture

  int   nBuf[] = {gr.nh[0], gr.nh[1], gr.nh[2]}; nBuf[myDir] = gr.h[myDir];
  int   nOff[] = {       0,        0,        0}; nOff[myDir] = gr.h[myDir];
  range<3> rBuf(nBuf[0], nBuf[1], nBuf[2]);
  Log::clog(8) << TAG << " Filling buffers ..." << Log::endl;

  //-- Wrap each field as a 3D mdspan over the full WH grid
  ms::mdspan<real, dex3> v_ms[MAX_FIELDS];
  for (int f = 0; f < nFields; ++f)
    v_ms[f] = ms::mdspan<real, dex3>(v[f], gr.nh[0], gr.nh[1], gr.nh[2]);

  //-- Helper: wrap a flat buffer as 4D mdspan (nFields × slab extents)
  auto buf4d = [=](real* raw, size_t e0, size_t e1, size_t e2) {
    return ms::mdspan<real, dex4>(raw, nFields, e0, e1, e2);
  };

  //-- Submdspan helpers for the 4 slabs (safe + reconstruct layout_right)
  auto full_ext = ms::full_extent;
  using pair = std::pair<size_t, size_t>;

  int h  = gr.h[myDir];
  int nhd = gr.nh[myDir];

  auto safe_sub = [&](ms::mdspan<real, dex3> const& f, pair p0, pair p1, pair p2) {
    auto sub = ms::submdspan(f, p0, p1, p2);
    return ms::mdspan<real, dex3>(sub.data_handle(), sub.extents());
  };

  // Grid coordinate at forward iteration (i, j, k) ∈ nBuf[0..2]
  // operation: 0=left_send, 1=right_send, 2=left_recv, 3=right_recv
  auto grid_pos = [=](int op, int id0, int id1, int id2) -> std::array<size_t, 3> {
    int idx[] = {id0, id1, id2};
    int pos[] = {0, 0, 0};
    for (int d = 0; d < 3; ++d) {
      if (d == myDir) {
        switch (op) {
          case 0: pos[d] = h + i0 + idx[d]; break;  // left send:  first h inner cells
          case 1: pos[d] = nhd - 2*h - i0 + idx[d]; break; // right send: last h inner cells
          case 2: pos[d] = idx[d]; break;             // left recv:  left halo
          case 3: pos[d] = nhd - h + idx[d]; break;  // right recv: right halo
        }
      } else {
        pos[d] = idx[d];
      }
    }
    return {static_cast<size_t>(pos[0]), static_cast<size_t>(pos[1]), static_cast<size_t>(pos[2])};
  };

  //-- MPI buffer start
  MPI_Start(&reqRecvL[myDir]);
  MPI_Start(&reqRecvR[myDir]);

  // All 4 operations share the same slab extents = nBuf
  size_t e0 = rBuf[0], e1 = rBuf[1], e2 = rBuf[2];
  auto r3 = range<3>(e0, e1, e2);

  //-- Pack LEFT send (first h inner cells → sendBufL)
  auto sbL = buf4d(sendBufL, e0, e1, e2);
  qq.parallel_for(r3, [=](item<3> it) {
    auto id = it.get_id();
    auto p  = grid_pos(0, id[0], id[1], id[2]);
    for (int f = 0; f < nFields; ++f)
      sbL(f, id[0], id[1], id[2]) = v_ms[f](p[0], p[1], p[2]);
  }).wait_and_throw();
  MPI_Start(&reqSendL[myDir]);

  //-- Pack RIGHT send (last h inner cells → sendBufR)
  auto sbR = buf4d(sendBufR, e0, e1, e2);
  qq.parallel_for(r3, [=](item<3> it) {
    auto id = it.get_id();
    auto p  = grid_pos(1, id[0], id[1], id[2]);
    for (int f = 0; f < nFields; ++f)
      sbR(f, id[0], id[1], id[2]) = v_ms[f](p[0], p[1], p[2]);
  }).wait_and_throw();

  //-- Middle communication
  MPI_Start(&reqSendR[myDir]);
  MPI_Wait(&reqRecvL[myDir], &status);
  Log::clog(8) << TAG << " Recopying from buffers..." << Log::endl;

  //-- Unpack RIGHT recv (bufL → right halo)
  auto bL4 = buf4d(bufL, e0, e1, e2);
  qq.parallel_for(r3, [=](item<3> it) {
    auto id = it.get_id();
    auto p  = grid_pos(3, id[0], id[1], id[2]);
    for (int f = 0; f < nFields; ++f)
      v_ms[f](p[0], p[1], p[2]) = bL4(f, id[0], id[1], id[2]);
  }).wait_and_throw();

  MPI_Wait(&reqRecvR[myDir], &status);

  //-- Unpack LEFT recv (bufR → left halo)
  auto bR4 = buf4d(bufR, e0, e1, e2);
  qq.parallel_for(r3, [=](item<3> it) {
    auto id = it.get_id();
    auto p  = grid_pos(2, id[0], id[1], id[2]);
    for (int f = 0; f < nFields; ++f)
      v_ms[f](p[0], p[1], p[2]) = bR4(f, id[0], id[1], id[2]);
  }).wait_and_throw();
  switch(bcType_[myDir]){ //-- PROCESSING BC TYPEs
    case BCOF0: //- Outflow w. 0th order interp
      Log::clog(10) << TAG << " Processing Outflow BCs of order 0..." << Log::endl;
      if(isEdgeLeft_[myDir]){
        int srcPos = gr.h[myDir] + i0;
        qq.parallel_for(r3, [=](item<3> it) {
          auto id = it.get_id();
          for (int f = 0; f < nFields; ++f)
            switch (myDir) {
              case 0: v_ms[f](id[0], id[1], id[2]) = v_ms[f](srcPos, id[1], id[2]); break;
              case 1: v_ms[f](id[0], id[1], id[2]) = v_ms[f](id[0], srcPos, id[2]); break;
              case 2: v_ms[f](id[0], id[1], id[2]) = v_ms[f](id[0], id[1], srcPos); break;
            }
        }).wait_and_throw();
      }
      if(isEdgeRight_[myDir]){
        int base = gr.nh[myDir] - gr.h[myDir];
        int srcPos = base - 1;
        qq.parallel_for(r3, [=](item<3> it) {
          auto id = it.get_id();
          for (int f = 0; f < nFields; ++f)
            switch (myDir) {
              case 0: v_ms[f](base+id[0], id[1], id[2]) = v_ms[f](srcPos, id[1], id[2]); break;
              case 1: v_ms[f](id[0], base+id[1], id[2]) = v_ms[f](id[0], srcPos, id[2]); break;
              case 2: v_ms[f](id[0], id[1], base+id[2]) = v_ms[f](id[0], id[1], srcPos); break;
            }
        }).wait_and_throw();
      }
      break;
    case BCOF3: //- Outflow w. 3rd order (cubic Lagrange extrapolation)
      Log::clog(10) << TAG << " Processing Outflow BCs of order 3..." << Log::endl;
      if (isEdgeLeft_[myDir]) {
        int h = gr.h[myDir];
        int src0 = h + i0;
        qq.parallel_for(r3, [=](item<3> it) {
          auto id = it.get_id();
          int g = id[myDir];
          int k = h - g;
          real c[4];
          switch (k) {
            case 1: c[0]=4; c[1]=-6; c[2]=4; c[3]=-1; break;
            case 2: c[0]=10; c[1]=-20; c[2]=15; c[3]=-4; break;
            case 3: c[0]=20; c[1]=-45; c[2]=36; c[3]=-10; break;
            case 4: c[0]=35; c[1]=-84; c[2]=70; c[3]=-20; break;
          }
          int p0 = src0, p1 = src0+1, p2 = src0+2, p3 = src0+3;
          for (int f = 0; f < nFields; ++f) {
            real val;
            switch (myDir) {
              case 0:
                val = c[0]*v_ms[f](p0,id[1],id[2]) + c[1]*v_ms[f](p1,id[1],id[2])
                    + c[2]*v_ms[f](p2,id[1],id[2]) + c[3]*v_ms[f](p3,id[1],id[2]);
                v_ms[f](g,id[1],id[2]) = val; break;
              case 1:
                val = c[0]*v_ms[f](id[0],p0,id[2]) + c[1]*v_ms[f](id[0],p1,id[2])
                    + c[2]*v_ms[f](id[0],p2,id[2]) + c[3]*v_ms[f](id[0],p3,id[2]);
                v_ms[f](id[0],g,id[2]) = val; break;
              case 2:
                val = c[0]*v_ms[f](id[0],id[1],p0) + c[1]*v_ms[f](id[0],id[1],p1)
                    + c[2]*v_ms[f](id[0],id[1],p2) + c[3]*v_ms[f](id[0],id[1],p3);
                v_ms[f](id[0],id[1],g) = val; break;
            }
          }
        }).wait_and_throw();
      }
      if (isEdgeRight_[myDir]) {
        int h = gr.h[myDir];
        int base = gr.nh[myDir] - h;
        qq.parallel_for(r3, [=](item<3> it) {
          auto id = it.get_id();
          int i = id[myDir];
          int k = i + 1;
          real c[4];
          switch (k) {
            case 1: c[0]=4; c[1]=-6; c[2]=4; c[3]=-1; break;
            case 2: c[0]=10; c[1]=-20; c[2]=15; c[3]=-4; break;
            case 3: c[0]=20; c[1]=-45; c[2]=36; c[3]=-10; break;
            case 4: c[0]=35; c[1]=-84; c[2]=70; c[3]=-20; break;
          }
          int p0 = base-1, p1 = base-2, p2 = base-3, p3 = base-4;
          for (int f = 0; f < nFields; ++f) {
            real val;
            switch (myDir) {
              case 0:
                val = c[0]*v_ms[f](p0,id[1],id[2]) + c[1]*v_ms[f](p1,id[1],id[2])
                    + c[2]*v_ms[f](p2,id[1],id[2]) + c[3]*v_ms[f](p3,id[1],id[2]);
                v_ms[f](base+i,id[1],id[2]) = val; break;
              case 1:
                val = c[0]*v_ms[f](id[0],p0,id[2]) + c[1]*v_ms[f](id[0],p1,id[2])
                    + c[2]*v_ms[f](id[0],p2,id[2]) + c[3]*v_ms[f](id[0],p3,id[2]);
                v_ms[f](id[0],base+i,id[2]) = val; break;
              case 2:
                val = c[0]*v_ms[f](id[0],id[1],p0) + c[1]*v_ms[f](id[0],id[1],p1)
                    + c[2]*v_ms[f](id[0],id[1],p2) + c[3]*v_ms[f](id[0],id[1],p3);
                v_ms[f](id[0],id[1],base+i) = val; break;
            }
          }
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
