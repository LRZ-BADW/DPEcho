//  Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//  Copyright(C) 2021 Salvatore Cielo, LRZ
//  Copyright(C) 2021 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
//  with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is
//  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and limitations under the License.

#include "Logger.hpp"
#include "Grid.hpp"
#include "Domain.hpp"
#include "Problem.hpp"
#include "Physics.hpp"
#include "Solver.hpp"
#include "Metric.hpp"
#include "Output.hpp"
#include "Device.hpp"
#include "Parameters.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>

using namespace sycl;

int main(int argc, char** argv ) {
  using namespace std::string_literals;
  //-- Load Parameter file
  std::string parFile = (argc > 1) ? argv[1] : "dpecho.par";
  Parameters param(parFile);

  //-- Logger
  int clogVerbosity = param.getOr("clogVerb", 4);
  int coutVerbosity = param.getOr("coutVerb", 4);
  std::string logfileName = param.getOr("clogName", "log"s);
  Log::init(logfileName, coutVerbosity, clogVerbosity);
  Log::togglePcontrol(0);

  //-- Parameters
  unsigned Mx = param.getOr("Mx", 24), My = param.getOr("My", 24), Mz = param.getOr("Mz", 24);
  unsigned Hx = param.getOr("Hx", 24), Hy = param.getOr("Hy", 24), Hz = param.getOr("Hz", 24);

  field bha = param.getOr<field>("bha", 0.4), bhm = param.getOr<field>("bhm", 0.25), bhc =param.getOr<field>("bhc", 0.0);

  bool dumpHalos = static_cast<bool>(param.getOr("dumpHalos", 0));

  Log::cout(0) << TAG << "Grid size: Mx: "<< Mx  <<", My: "<<My  <<", Mz: "<<Mz << Log::endl;

  //-- Further parameter processing
  if(Hx<NGC){ Hx=NGC; Log::cout(4) << "Raising Hx to allowed min " << NGC << Log::endl; }
  if(Hy<NGC){ Hy=NGC; Log::cout(4) << "Raising Hy to allowed min " << NGC << Log::endl; }
  if(Hz<NGC){ Hz=NGC; Log::cout(4) << "Raising Hz to allowed min " << NGC << Log::endl; }
  Metric::setParameters(bha, bhm, bhc);   // Only relevant for non-Cartesian metrics, otherwise, no-op.

  //-- SYCL device selection
  Device dc; // automation inside Deviceparam
  sycl::queue qDev(dc.deviceWith(param));
  const size_t  gMax = qDev.get_device().get_info<sycl::info::device::max_work_group_size>();
  const size_t wgMax = param.getOr<int>("wgMax", 4); // Safe and performant default, tune at will.

  //-- Sizes, Domain, Grids
  size_t bufSizes[] = {(Mx+2*Hx)*(My+2*Hy)*Hz,(My+2*Hy)*(Mz+2*Hz)*Hx,(Mz+2*Hz)*(Mx+2*Hx)*Hy}; // BCex buffer size for the largest case
  Domain *DD = new Domain(qDev, bufSizes, param);  DD->boxInfo();  DD->locInfo();
  Grid grid    = Grid(Mx  ,My  ,Mz  ,Hx  ,Hy  ,Hz  , DD->locMin(0),DD->locMax(0), DD->locMin(1),DD->locMax(1), DD->locMin(2),DD->locMax(2));
  Grid gridF[3]={Grid(Mx+1,My  ,Mz  ,Hx-2,   0,   0, 0.,1., 0.,1., 0.,1.),
                 Grid(Mx  ,My+1,Mz  ,   0,Hy-2,   0, 0.,1., 0.,1., 0.,1.),
                 Grid(Mx  ,My  ,Mz+1,   0,   0,Hz-2, 0.,1., 0.,1., 0.,1.)};
  grid.print();  for (int i=0; i<3; ++i){ gridF[i].print();}
  unsigned Ncell = grid.nht, Nout = dumpHalos ? Ncell : grid.nt; // For comfort
  unsigned Mmax  = std::max({Mx,My,Mz}),  Nflux = Mx*My*Mz/Mmax*(Mmax+1+2*(NGC-1)); // Flux have +1 point

  // -- Allocations
  int ok = 1;
  field *out[FLD_TOT]; for (int i=0; i<FLD_TOT; ++i){out[i] = malloc_shared<field>(Nout , qDev); ok*=(NULL!=out[i]); } // For ease of custom output
  field   *v[FLD_TOT]; for (int i=0; i<FLD_TOT; ++i){  v[i] = malloc_device<field>(Ncell, qDev); ok*=(NULL!=  v[i]); } // Primitives
  field   *u[FLD_TOT]; for (int i=0; i<FLD_TOT; ++i){  u[i] = malloc_device<field>(Ncell, qDev); ok*=(NULL!=  u[i]); } // Conserved
  field  *du[FLD_TOT]; for (int i=0; i<FLD_TOT; ++i){ du[i] = malloc_device<field>(Ncell, qDev); ok*=(NULL!= du[i]); } // Time Evolution
  field  *u0[FLD_TOT]; for (int i=0; i<FLD_TOT; ++i){ u0[i] = malloc_device<field>(Ncell, qDev); ok*=(NULL!= u0[i]); } // RK basis
  field   *f[FLD_TOT]; for (int i=0; i<FLD_TOT; ++i){  f[i] = malloc_device<field>(Nflux, qDev); ok*=(NULL!=  f[i]); } // Fluxes
#ifdef UCT
  field *apG[3];  for (int i=0; i<3; ++i){ apG[i] = malloc_device<field>(Nflux, qDev); ok *= (NULL!=apG[i]); } // FWD characteristics (best with CT)
  field *amG[3];  for (int i=0; i<3; ++i){ amG[i] = malloc_device<field>(Nflux, qDev); ok *= (NULL!=amG[i]); } // BWD characteristics (best with CT)
  field *vt0[3];  for (int i=0; i<3; ++i){ vt0[i] = malloc_device<field>(Nflux, qDev); ok *= (NULL!=vt0[i]); } // Transverse vel. 0
  field *vt1[3];  for (int i=0; i<3; ++i){ vt1[i] = malloc_device<field>(Nflux, qDev); ok *= (NULL!=vt1[i]); } // Transverse vel. 1
#endif
#ifndef NDEBUG    // For printing arbitrary intermediate values
  field *debug[FLD_TOT];for (int i=0; i < FLD_TOT; ++i){debug[i] = malloc_shared<field>(Ncell, qDev); ok *= (NULL!=debug[i]); }
#endif
  Log::Assert(ok, "Cannot allocate data. Exiting");

  //-- Problem
  field dtLoc; // local copy of time, for ease of capture
  Problem problem(qDev, param, &grid, DD, out);
  problem.init(v, u);  // Inits v and u in DEVICE based on param scenario, calls BCex, prints ICs.

  //-- SYCL ranges and related accessories
  range<3> rStd  = range(grid.n[0], grid.n[1], grid.n[2]);
  field   *aMax  = malloc_shared<field>(3, qDev), vChar; // For reduction, and CFL in timestepping

  // Main Evolution loop
  Log::togglePcontrol(1); // start profiling
  while( (problem.t() <= problem.tMax()) && (problem.iStep() < problem.nStep()) ){

    for (int irk = 0; irk < NRK; irk++){  // RK loop
      if (!irk){ aMax[0]=0.0; aMax[1]=0.0; aMax[2]=0.0; }

      for(unsigned myDir=0; myDir<3; myDir++){ // Direction loop
        //range<3> rLoc(gridF[myDir].groupSize[0], gridF[myDir].groupSize[1], gridF[myDir].groupSize[2]);
        auto maxReduction = sycl::reduction(aMax + myDir, sycl::maximum<field>());

        //-- Flux kernel (PoV of f[])
        range<3> rFlx = range(gridF[myDir].n[0], gridF[myDir].n[1], gridF[myDir].n[2]); // Fluxes along this direction
        //-- Nameless kernels as sometimes name and reduction clash (eg. AMD with LLVM-Intel)
	qDev.parallel_for(getMatchingNdRange(rFlx, range<3>(wgMax,wgMax,wgMax)), maxReduction, [=](nd_item<3> it, auto &max) {
          //-- Several varied indexes and stuff... SYCL USM is not ready for this!
          id<3> gid = it.get_global_id();  // Flux indexes
          if (isOutOfBounds(gid, rFlx)){ return; } // Allows for arbitrary grid and workgroup sizes. Circumvents nvidia bug.
          int vOff[]={grid.h        [0] ,grid.h        [1], grid.h        [2]};  vOff[myDir]+= -1 ;
          int fOff[]={gridF[myDir].h[0], gridF[myDir].h[1], gridF[myDir].h[2]};
          int fId=globLinId(gid,gridF[myDir].nh, fOff), fSt=stride(gid, myDir, gridF[myDir].nh); // Accessing fluxes
          int vId=globLinId(gid,grid.nh        , vOff), vSt=stride(gid, myDir, grid.nh        ); // Accessing v, u

          // What you declare here resides in GPU core-memory - SC
          field vR[FLD_TOT],vL[FLD_TOT]; for (int i=0; i<FLD_TOT; ++i){ holibRec(vId,v[i],vSt,vL+i,vR+i); }
          Metric g(grid.xC(gid, 0), grid.xC(gid, 1), grid.xC(gid, 2));
          field uR[FLD_TOT],fR[FLD_TOT],vfR[2],vtR[2];  physicalFlux(myDir, g, vR, uR, fR, vfR, vtR);
          field uL[FLD_TOT],fL[FLD_TOT],vfL[2],vtL[2];  physicalFlux(myDir, g, vL, uL, fL, vfL, vtL);
          field ap = sycl::max((field)0., sycl::max( vfL[0], vfR[0]));
          field am = sycl::max((field)0., sycl::max(-vfL[1],-vfR[1]));
#ifdef UCT // For induction we save these too
          apG[myDir][fId] = ap;  vt1[myDir][fId] = (ap*vtL[0]+am*vtR[0])/(ap+am);
          amG[myDir][fId] = am;  vt2[myDir][fId] = (ap*vtL[1]+am*vtR[1])/(ap+am);
#endif
          // Fluxes from reconstructed values. When CT is on, this loop leaves B fields out
          for (int i=0; i<FLD_TOT; ++i){ f[i][fId] = (ap*fL[i]+am*fR[i]-ap*am*(uR[i]-uL[i]))/(ap+am); }

          // For timestepping; needed only if 0==irk
          if(!irk){field localMax = sycl::max(ap, am);  max.combine(localMax); }

          if (!myDir){ //- Source terms. Do it once per du calculation (TODO: is this right with the RK? check!)
            for (int i=0; i<FLD_TOT; ++i){ du[i][vId] = 0.0; };
            field src[4]; physicalSource(vId, v, g, src);
            du[VX][vId] =-src[0]; du[VY][vId] =-src[1]; du[VZ][vId] =-src[2]; du[PG][vId] =-src[3];
          }

#ifndef NDEBUG  // Variables you may print for debug. set debug:= <whatYouWantToSee>
          for (int i=0; i<FLD_TOT; ++i){ debug[i][vId] = vL[i]; }
          // Some examples for what you may want to debug!
          debug[0][vId] = g.gCon(0,0); debug[1][vId] = g.gCon(1,1);
          debug[2][vId] = g.gCon(2,2); debug[3][vId] = g.gCon(0,2);
          debug[4][vId] = uR[0]; debug[5][vId] =-uL[0]; debug[6][vId] = uR[0]-uL[0];
#endif
        }); // End parallel_for
        qDev.wait_and_throw(); // Now the flux is available everywhere

        //-- Flux reconstr & Derivatives. Trying to provide more halos to skip this was not beneficial!
        DD->BCex(myDir, gridF[myDir], f, BCEX_FL); // call BCEX on fluxes.

#ifndef NDEBUG
	problem.dump(f    , gridF[myDir], "flux"+std::to_string(myDir));
	problem.dump(debug, grid        , "debug"                     );
#endif
        qDev.parallel_for<class parForUpdateDu>(rStd, [=](item<3> it) { //-- Update du with current direction
          id<3> id = it.get_id();
          if (isOutOfBounds(id, rStd)){ return; }
          int myId   = globLinId(id, grid.nh        , grid.h        ); // Accessing v, u and the like
          int fId    = globLinId(id, gridF[myDir].nh, gridF[myDir].h); // Accessing fluxes
          int dStride= stride   (id,   myDir, gridF[myDir].nh); // Byproduct of the above
          for (int i=0; i<FLD_TOT; ++i){ du[i][myId]+= holibDer(fId, f[i], dStride)/grid.dx[myDir];}
        }).wait_and_throw(); // Now we have the du up to the current direction
      } // End loop on directions

      if (0 == irk){ //- Only at the end of 1st RK step compute the timestep & print time (less MPI barriers)
        problem.lap(); // Store the timestep value before barrier, to estimate load imbalance.
#ifdef MPICODE
        MPI_Allreduce(MPI_IN_PLACE, aMax, 3, MPI_FIELD, MPI_MAX, MPI_COMM_WORLD ); // MPI_COMM_WORLD is an epsilon faster than DD->cartComm()
#endif
        vChar=std::max( {aMax[0]/grid.dx[0], aMax[1]/grid.dx[1], aMax[2]/grid.dx[2]} );  // Accumulation
        problem.dtUpdate(vChar); dtLoc = problem.dt(); // Update timing & print it
        for (int i=0; i<FLD_TOT; ++i) { qDev.memcpy(u0[i], u[i], Ncell*sizeof(field)); } // Store original u: u0 = u
        qDev.wait_and_throw();
      }

      qDev.parallel_for<class parForRK>(rStd, [=](item<3> it) { //-- Updating RK
        id<3> id = it.get_id();
        range<3> ar = it.get_range();
        if (isOutOfBounds(id, rStd)){ return; }
        int myId = globLinId(it.get_id(), grid.nh, grid.h ); // Accessing v, u and the like
        for (int i=0; i<FLD_TOT; ++i)
          u[i][myId] = crk1[irk] * u0[i][myId] + crk2[irk]*( u[i][myId] - dtLoc*du[i][myId] );
        Metric g(grid.xC(id, 0), grid.xC(id, 1), grid.xC(id, 2));
        cons2prim(myId, Ncell, u, v, g);
      }).wait_and_throw();

      for(unsigned myDir=0; myDir<3; myDir++) { DD->BCex(myDir, grid, v); }

    }//-- END RK

    // Log timestep report
    Log::clog(4) << TAG<<"Step# "<< problem.iStep()<<", dump# "<< problem.iOut()-1 << ", characteristic "<< vChar << Log::endl;
#ifndef NDEBUG
    problem.dump(u);
#endif
    if( problem.t() / problem.tMax() > problem.iOut() * problem.tOut() ){
      Log::togglePcontrol(0); // When profiling, exclude output
      problem.dump(v);
      Log::togglePcontrol(1);
    }
  } // Evolution while

  Log::togglePcontrol(0);

  for (int i=0; i < FLD_TOT; ++i){ free(out[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(  v[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(  u[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free( u0[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(  f[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free( du[i], qDev); }
#ifdef UCT
  for (int i=0; i < 3; ++i){ free(  apG[i], qDev); }
  for (int i=0; i < 3; ++i){ free(amGdu[i], qDev); }
  for (int i=0; i < 3; ++i){ free(  vt1[i], qDev); }
  for (int i=0; i < 3; ++i){ free(  vt2[i], qDev); }
#endif
#ifndef NDEBUG
  for (int i=0; i < FLD_TOT; ++i){ free(debug[i], qDev); }
#endif
  Log::cout() << problem.getTimings() << Log::endl;
  Log::finalize();
  return 0;
} // end main
