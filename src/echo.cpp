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
#include "Device.hpp"
#include "Parameters.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <ctime>
#include <filesystem>

using namespace sycl;

int main(int argc, char** argv ) {
  using namespace std::string_literals;
  //-- Load Parameter file
  std::string parFile = (argc > 1) ? argv[1] : "dpecho.par";
  Parameters param(parFile);

  //-- Logger
  std::string runName = param.getOr("runName", std::filesystem::path(parFile).stem().string());
  std::string restartDir = param.getOr("restartDir", ""s);
  if (!restartDir.empty()) {
    runName = restartDir + "/" + runName;
  } else {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H%M_") << runName;
    runName = ss.str();
  }
  int verbosity = param.getOr("verbosity", 4);
  Log::init(runName + "/log", verbosity);
  Log::cout(2) << TAG << "Logger initialized with verbosity=" << verbosity << Log::endl;
  Log::togglePcontrol(0);

  //-- Physics selection
  std::string physType = param.getOr("physics", "GRMHD"s);
  Log::cout(7) << TAG << "Physics type: " << physType << Log::endl;

  //-- Parameters
  unsigned Mx = param.getOr("Mx", 24), My = param.getOr("My", 24), Mz = param.getOr("Mz", 24);
  unsigned Hx = param.getOr("Hx", 24), Hy = param.getOr("Hy", 24), Hz = param.getOr("Hz", 24);

  real bha = param.getOr<real>("bha", 0.4), bhm = param.getOr<real>("bhm", 0.25), bhc =param.getOr<real>("bhc", 0.0);

  bool dumpHalos = static_cast<bool>(param.getOr("dumpHalos", 0));

  Log::cout(0) << TAG << "Grid size: Mx: "<< Mx  <<", My: "<<My  <<", Mz: "<<Mz << Log::endl;

  //-- Further parameter processing
  if(Hx<NGC){ Hx=NGC; Log::cout(4) << "Raising Hx to allowed min " << NGC << Log::endl; }
  if(Hy<NGC){ Hy=NGC; Log::cout(4) << "Raising Hy to allowed min " << NGC << Log::endl; }
  if(Hz<NGC){ Hz=NGC; Log::cout(4) << "Raising Hz to allowed min " << NGC << Log::endl; }
  Metric::setParameters(bha, bhm, bhc);   // Only relevant for non-Cartesian metrics, otherwise, no-op.

  //-- SYCL device selection
  Device dc;
  sycl::queue qDev(dc.deviceWith(param));
  const size_t  gMax = qDev.get_device().get_info<sycl::info::device::max_work_group_size>();
  const size_t wgMax = param.getOr<int>("wgMax", 4); // Safe and performant default, tune at will.
  Log::cout(6) << TAG << "Device selected, max_work_group_size=" << gMax << ", wgMax=" << wgMax << Log::endl;

  //-- Physics on device
  Physics *phys = sycl::malloc_shared<Physics>(1, qDev);
  new(phys) Physics(physType);
  int nFields = phys->fldTot();
  Log::cout(7) << TAG << "Physics object created: " << physType << ", nFields=" << nFields << Log::endl;

  //-- Sizes, Domain, Grids
  size_t bufSizes[NDIM]={(Mx+2*Hx)*(My+2*Hy)*Hz,(My+2*Hy)*(Mz+2*Hz)*Hx,(Mz+2*Hz)*(Mx+2*Hx)*Hy}; // BCex buffer size for the largest case
  Domain *DD = new Domain(qDev, bufSizes, param, nFields);  DD->boxInfo();  DD->locInfo();
  Log::cout(7) << TAG << "Domain created, grid ready" << Log::endl;
  Grid grid    = Grid(Mx  ,My  ,Mz  ,Hx  ,Hy  ,Hz  , DD->locMin(0),DD->locMax(0), DD->locMin(1),DD->locMax(1), DD->locMin(2),DD->locMax(2));
  Grid gridF[NDIM]={Grid(Mx+1,My  ,Mz  ,Hx-2,   0,   0, 0.,1., 0.,1., 0.,1.),
                 Grid(Mx  ,My+1,Mz  ,   0,Hy-2,   0, 0.,1., 0.,1., 0.,1.),
                 Grid(Mx  ,My  ,Mz+1,   0,   0,Hz-2, 0.,1., 0.,1., 0.,1.)};
  grid.print();  for (int i=0; i<NDIM; ++i){ gridF[i].print();}
  Log::cout(8) << TAG << "Grid layout printed" << Log::endl;
  unsigned Ncell = grid.nht, Nout = dumpHalos ? Ncell : grid.nt; // For comfort
  unsigned Mmax  = std::max({Mx,My,Mz}),  Nflux = Mx*My*Mz/Mmax*(Mmax+1+2*(NGC-1)); // Flux have +1 point

  // -- Allocations
  int ok = 1;
  real *out = malloc_shared<real>(Nout * nFields, qDev); ok = (NULL != out);
  real **v  = new real*[nFields]; for (int i=0; i<nFields; ++i){  v[i] = malloc_device<real>(Ncell, qDev); ok*=(NULL!=  v[i]); } // Primitives
  real **u  = new real*[nFields]; for (int i=0; i<nFields; ++i){  u[i] = malloc_device<real>(Ncell, qDev); ok*=(NULL!=  u[i]); } // Conserved
  real **du = new real*[nFields]; for (int i=0; i<nFields; ++i){ du[i] = malloc_device<real>(Ncell, qDev); ok*=(NULL!= du[i]); } // Time Evolution
  real **u0 = new real*[nFields]; for (int i=0; i<nFields; ++i){ u0[i] = malloc_device<real>(Ncell, qDev); ok*=(NULL!= u0[i]); } // RK basis
  real **f  = new real*[nFields]; for (int i=0; i<nFields; ++i){  f[i] = malloc_device<real>(Nflux, qDev); ok*=(NULL!=  f[i]); } // Fluxes
#ifdef UCT
  real *apG[NDIM];  for (int i=0; i<NDIM; ++i){ apG[i] = malloc_device<real>(Nflux, qDev); ok *= (NULL!=apG[i]); } // FWD characteristics (best with CT)
  real *amG[NDIM];  for (int i=0; i<NDIM; ++i){ amG[i] = malloc_device<real>(Nflux, qDev); ok *= (NULL!=amG[i]); } // BWD characteristics (best with CT)
  real *vt0[NDIM];  for (int i=0; i<NDIM; ++i){ vt0[i] = malloc_device<real>(Nflux, qDev); ok *= (NULL!=vt0[i]); } // Transverse vel. 0
  real *vt1[NDIM];  for (int i=0; i<NDIM; ++i){ vt1[i] = malloc_device<real>(Nflux, qDev); ok *= (NULL!=vt1[i]); } // Transverse vel. 1
#endif
#ifndef NDEBUG    // For printing arbitrary intermediate values
  real **debug = new real*[nFields];for (int i=0; i < nFields; ++i){debug[i] = malloc_shared<real>(Ncell, qDev); ok *= (NULL!=debug[i]); }
#endif
  Log::Assert(ok, "Cannot allocate data. Exiting");
  Log::cout(7) << TAG << "Memory allocations completed" << Log::endl;

  //-- Problem
  real dtLoc; // local copy of time, for ease of capture
  Problem problem(qDev, param, &grid, DD, phys, out, runName);
  Log::cout(7) << TAG << "Problem initialized" << Log::endl;
  if (!restartDir.empty()) {
    problem.restart(v, u, restartDir);
  } else {
    problem.init(v, u);  // Inits v and u in DEVICE based on param scenario, calls BCex, prints ICs.
  }

  //-- SYCL ranges and related accessories
  range<3> rStd  = range(grid.n[0], grid.n[1], grid.n[2]);
  real *aMax  = malloc_shared<real>(NDIM, qDev), vChar; // For reduction, and CFL in timestepping
  // Main Evolution loop
  Log::togglePcontrol(1); // start profiling
  Log::cout(6) << TAG << "Starting evolution loop" << Log::endl;
  while( (problem.t() <= problem.tMax()) && (problem.iStep() < problem.nStep()) ){

    for (int irk = 0; irk < NRK; irk++){  // RK loop
      Log::cout(8) << TAG << "RK iteration " << irk << " start" << Log::endl;
      if (!irk){ aMax[0]=0.0; aMax[1]=0.0; aMax[2]=0.0; }

      for(unsigned myDir=0; myDir<NDIM; myDir++){ // Direction loop
        Log::cout(8) << TAG << "Direction " << myDir << " start" << Log::endl;
        auto maxReduction = sycl::reduction(aMax + myDir, sycl::maximum<real>());

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
          real vR[MAX_FIELDS],vL[MAX_FIELDS]; for (int i=0; i<nFields; ++i){ holibRec(vId,v[i],vSt,vL+i,vR+i); }
          Metric g(grid.xC(gid, 0), grid.xC(gid, 1), grid.xC(gid, 2));
          real uR[MAX_FIELDS],fR[MAX_FIELDS],vfR[2],vtR[2];  phys->physicalFlux(myDir, g, vR, uR, fR, vfR, vtR);
          real uL[MAX_FIELDS],fL[MAX_FIELDS],vfL[2],vtL[2];  phys->physicalFlux(myDir, g, vL, uL, fL, vfL, vtL);
          real ap = sycl::max((real)0., sycl::max( vfL[0], vfR[0]));
          real am = sycl::max((real)0., sycl::max(-vfL[1],-vfR[1]));
#ifdef UCT // For induction we save these too
          apG[myDir][fId] = ap;  vt1[myDir][fId] = (ap*vtL[0]+am*vtR[0])/(ap+am);
          amG[myDir][fId] = am;  vt2[myDir][fId] = (ap*vtL[1]+am*vtR[1])/(ap+am);
#endif
          // Fluxes from reconstructed values. When CT is on, this loop leaves B reals out
          real apam = ap + am; if (apam <= 0) { apam = 1e-30; }
          for (int i=0; i<nFields; ++i){ real flx = (ap*fL[i]+am*fR[i]-ap*am*(uR[i]-uL[i]))/apam; f[i][fId] = (flx==flx) ? flx : 0; }

          // For timestepping; needed only if 0==irk
          if(!irk){real localMax = sycl::max(ap, am);  max.combine(localMax); }

          if (!myDir){ //- Source terms. Do it once per du calculation
            for (int i=0; i<nFields; ++i){ du[i][vId] = 0.0; };
            real src[4]; phys->physicalSource(vId, v, g, src);
            du[VX][vId] = (src[0]==src[0]) ? -src[0] : 0;
            du[VY][vId] = (src[1]==src[1]) ? -src[1] : 0;
            du[VZ][vId] = (src[2]==src[2]) ? -src[2] : 0;
            du[PG][vId] = (src[3]==src[3]) ? -src[3] : 0;
          }

#ifndef NDEBUG  // Variables you may print for debug. set debug:= <whatYouWantToSee>
          for (int i=0; i<nFields; ++i){ debug[i][vId] = vL[i]; }
          // Some examples for what you may want to debug!
          debug[0][vId] = g.gCon(0,0); debug[1][vId] = g.gCon(1,1);
          debug[2][vId] = g.gCon(2,2); debug[3][vId] = g.gCon(0,2);
          debug[4][vId] = uR[0]; debug[5][vId] =-uL[0]; debug[6][vId] = uR[0]-uL[0];
#endif
        }); // End parallel_for
        qDev.wait_and_throw(); // Now the flux is available everywhere

        //-- Flux reconstr & Derivatives. Trying to provide more halos to skip this was not beneficial!
        DD->BCex(myDir, gridF[myDir], f, BCEX_FL); // call BCEX on fluxes.
        Log::cout(9) << TAG << "BCex completed for direction " << myDir << " (fluxes)" << Log::endl;

#ifndef NDEBUG
	problem.dump(f    , gridF[myDir], runName+"/flux"+std::to_string(myDir), runName);
	problem.dump(debug, grid        , runName+"/debug"                     , runName);
#endif
        qDev.parallel_for<class parForUpdateDu>(rStd, [=](item<3> it) { //-- Update du with current direction
          id<3> id = it.get_id();
          if (isOutOfBounds(id, rStd)){ return; }
          int myId   = globLinId(id, grid.nh        , grid.h        ); // Accessing v, u and the like
          int fId    = globLinId(id, gridF[myDir].nh, gridF[myDir].h); // Accessing fluxes
          int dStride= stride   (id,   myDir, gridF[myDir].nh); // Byproduct of the above
          for (int i=0; i<nFields; ++i){ du[i][myId]+= holibDer(fId, f[i], dStride)/grid.dx[myDir];}
        }).wait_and_throw(); // Now we have the du up to the current direction
      } // End loop on directions
      Log::cout(9) << TAG << "Lap completed for direction loop" << Log::endl;

      if (0 == irk){ //- Only at the end of 1st RK step compute the timestep & print time (less MPI barriers)
        problem.lap(); // Store the timestep value before barrier, to estimate load imbalance.
        MPI_Allreduce(MPI_IN_PLACE, aMax, NDIM, MPI_REAL, MPI_MAX, MPI_COMM_WORLD ); // MPI_COMM_WORLD is an epsilon faster than DD->cartComm()
        vChar=std::max( {aMax[0]/grid.dx[0], aMax[1]/grid.dx[1], aMax[2]/grid.dx[2]} );  // Accumulation
        problem.dtUpdate(vChar); dtLoc = problem.dt(); // Update timing & print it
        for (int i=0; i<nFields; ++i) { qDev.memcpy(u0[i], u[i], Ncell*sizeof(real)); } // Store original u: u0 = u
        qDev.wait_and_throw();
      }

      {
        real c2pTol = problem.tolCons2Prim(); // tighter after restart
        qDev.parallel_for<class parForRK>(rStd, [=](item<3> it) { //-- Updating RK
          id<3> id = it.get_id();
          range<3> ar = it.get_range();
          if (isOutOfBounds(id, rStd)){ return; }
          int myId = globLinId(it.get_id(), grid.nh, grid.h ); // Accessing v, u and the like
          for (int i=0; i<nFields; ++i)
            u[i][myId] = crk1[irk] * u0[i][myId] + crk2[irk]*( u[i][myId] - dtLoc*du[i][myId] );
          u[RH][myId] = sycl::max(u[RH][myId], (real)RHOFLOOR);
          Metric g(grid.xC(id, 0), grid.xC(id, 1), grid.xC(id, 2));
          phys->cons2prim(myId, Ncell, u, v, g, c2pTol);
        }).wait_and_throw();
      }

      for(unsigned myDir=0; myDir<NDIM; myDir++) { DD->BCex(myDir, grid, v); }

    }//-- END RK
    qDev.wait_and_throw();
    // Log timestep report
    Log::cout(8) << TAG << "Timestep report: step=" << problem.iStep() << ", dt=" << problem.dt() << Log::endl;
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

  problem.waitOut();
  phys->~Physics(); sycl::free(phys, qDev);
  free(out, qDev);
  for (int i=0; i < nFields; ++i){ free(  v[i], qDev); } delete[] v;
  for (int i=0; i < nFields; ++i){ free(  u[i], qDev); } delete[] u;
  for (int i=0; i < nFields; ++i){ free( u0[i], qDev); } delete[] u0;
  for (int i=0; i < nFields; ++i){ free(  f[i], qDev); } delete[] f;
  for (int i=0; i < nFields; ++i){ free( du[i], qDev); } delete[] du;
#ifdef UCT
  for (int i=0; i < NDIM; ++i){ free(  apG[i], qDev); }
  for (int i=0; i < NDIM; ++i){ free(amGdu[i], qDev); }
  for (int i=0; i < NDIM; ++i){ free(  vt1[i], qDev); }
  for (int i=0; i < NDIM; ++i){ free(  vt2[i], qDev); }
#endif
#ifndef NDEBUG
  for (int i=0; i < nFields; ++i){ free(debug[i], qDev); } delete[] debug;
#endif
  Log::cout() << problem.getTimings() << Log::endl;
  Log::finalize();
  return 0;
} // end main
