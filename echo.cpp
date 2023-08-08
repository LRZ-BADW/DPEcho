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

#include <iostream>
#include <fstream>
#include "Logger.hpp"
#include "Grid.hpp"
#include "Domain.hpp"
#include "Problem.hpp"
#include "Physics.hpp"
#include "Metric.hpp"
#include "Solver.hpp"
#include "Output.hpp"
#include "DeviceConfig.hpp"
#include "echoSycl.hpp"

int main(int argc, char** argv ) {

  //-- Logger
  Logger *Log = Logger::getInstance( &argc, &argv );
  Log->togglePcontrol(0);
  Log->setInfoVerbosity(10); Log->setPar(false);

  //-- Parameters
  unsigned Mx = 24, My = 24,  Mz = 24, Hx = 4, Hy = 4, Hz = 4, nD = -1;
  int    logVerb = 4; bool dumpHalos = false;
#if SYCL!=oneAPI
  unsigned long locSize = 8;
#endif
  Log->setPar(false);   *Log+0<<TAG<<" Reading input: ";
  string confFile = (argc > 1) ? argv[1] : "echo.par";
  string key, val;  ifstream inFile(confFile);
  field bha = 0.0, bhm = 0.0, bhc = 0.0;
  while (getline(inFile, key, ' ') && getline(inFile, val)){
    if(!key.compare("Mx"       )){ Mx = stoi(val); *Log<<"\n\tMx "<< Mx ; continue;}
    if(!key.compare("My"       )){ My = stoi(val); *Log<<"\n\tMy "<< My ; continue;}
    if(!key.compare("Mz"       )){ Mz = stoi(val); *Log<<"\n\tMz "<< Mz ; continue;}
    if(!key.compare("Hx"       )){ Hx = stoi(val); *Log<<"\n\tHx "<< Hx ; continue;}
    if(!key.compare("Hy"       )){ Hy = stoi(val); *Log<<"\n\tHy "<< Hy ; continue;}
    if(!key.compare("Hz"       )){ Hz = stoi(val); *Log<<"\n\tHz "<< Hz ; continue;}
#if SYCL!=oneAPI
    if(!key.compare("locSize"  )){ locSize  = (size_t)stoi(val); *Log<<"\n\tlocSize   " << locSize   ; continue;}
#endif
    if(!key.compare("logVerb"  )){ logVerb  =         stoi(val); *Log<<"\n\tlogVerb   " << logVerb   ; continue;}
    if(!key.compare("dumpHalos")){ dumpHalos= (bool)  stoi(val); *Log<<"\n\tdumpHalos " << dumpHalos ; continue;}
    if(!key.compare("bha"      )){ bha = static_cast<field>(stod(val)); *Log<<"\n\tbha " << bha ; continue;}
    if(!key.compare("bhm"      )){ bhm = static_cast<field>(stod(val)); *Log<<"\n\tbhm " << bhm ; continue;}
    if(!key.compare("bhc"      )){ bhc = static_cast<field>(stod(val)); *Log<<"\n\tbhc " << bhc ; continue;}
    if(!key.compare("forceDevice")){ nD = stoi(val); *Log<<"\n\tnD "<< nD ; continue;}
  }
  //-- Further parameter processing
  Log->fl();  Log->setInfoVerbosity(logVerb);  Log->setPar(false);
  if(Hx<NGC){ Hx=NGC; Log->Info(4, "Raising Hx to allowed min %d\n", NGC ); }
  if(Hy<NGC){ Hy=NGC; Log->Info(4, "Raising Hy to allowed min %d\n", NGC ); }
  if(Hz<NGC){ Hz=NGC; Log->Info(4, "Raising Hz to allowed min %d\n", NGC ); }
  Metric::setParameters(bha, bhm, bhc);   // Only relevant for non-Cartesian metrics, otherwise, no-op.

  //-- SYCL device selection
  DeviceConfig dc; // automation inside DeviceConfig
  mysycl::queue qDev(dc.deviceWith(nD));

  // -- Domain & Grid
  Log->setPar(true);
  size_t bufMax = std::max({(Mx+2*Hx)*(My+2*Hy)*Hz,(My+2*Hy)*(Mz+2*Hz)*Hx,(Mz+2*Hz)*(Mx+2*Hx)*Hy}); // BCex buffer size for the largest case
  Domain *DD = new Domain(qDev, bufMax);  DD->boxInfo();  DD->locInfo();
  Grid grid = Grid(Mx,My,Mz, Hx,Hy,Hz, DD->locMin(0),DD->locMax(0), DD->locMin(1),DD->locMax(1), DD->locMin(2),DD->locMax(2));
  grid.print();
  unsigned Ncell = grid.nht, Nout = dumpHalos ? Ncell : grid.nt; // For comfort
  const int strides[3] = {stride(0,grid.nh),stride(1,grid.nh),stride(2,grid.nh)};
  
  // Used for mapping 3D grid to 1D grid
  const int start_idx = grid.h[0]*strides[0] + grid.h[1]*strides[1] + grid.h[2]*strides[2];
  const int stop_idx = (grid.n[0]+grid.h[0])*strides[0]+(grid.n[1]+grid.h[1])*strides[1]+(grid.n[2]+grid.h[2])*strides[2];

  // -- Allocations
  int ok = 1;
  field *out[FLD_TOT];  for (int i=0; i < FLD_TOT; ++i){out[i] = malloc_shared<field>(Nout , qDev); ok *= (NULL!=out[i]); } // For ease of custom output
  field   *v[FLD_TOT];  for (int i=0; i < FLD_TOT; ++i){  v[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=v[i]); } // Primitives
  field   *u[FLD_TOT];  for (int i=0; i < FLD_TOT; ++i){  u[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=u[i]); } // Conserved
  field   *f[FLD_TOT];  for (int i=0; i < FLD_TOT; ++i){  f[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=f[i]); } // Fluxes
  field  *du[FLD_TOT];  for (int i=0; i < FLD_TOT; ++i){ du[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=du[i]); } // Time Evolution
  field  *u0[FLD_TOT];  for (int i=0; i < FLD_TOT; ++i){ u0[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=u0[i]); } // RK basis
#ifndef NDEBUG    // For printing arbitrary intermediate values
  field *debug[FLD_TOT];for (int i=0; i < FLD_TOT; ++i){debug[i] = malloc_shared<field>(Ncell, qDev); ok *= (NULL!=debug[i]); }
#endif
#ifdef UCT
  field *apG[3];  for (int i=0; i < 3; ++i){ apG[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=apG[i]); } // FWD characteristics (best with CT)
  field *amG[3];  for (int i=0; i < 3; ++i){ amG[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=amG[i]); } // BWD characteristics (best with CT)
  field *vt0[3];  for (int i=0; i < 3; ++i){ vt0[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=vt0[i]); } // Transverse vel. 1
  field *vt1[3];  for (int i=0; i < 3; ++i){ vt1[i] = malloc_device<field>(Ncell, qDev); ok *= (NULL!=vt1[i]); } // Transverse vel. 1
#endif
  if(!ok){Log->Error("%s Cannot allocate data. Exiting.", TAG );}

  // -- Initialization
  Log->setPar(true);
  field dtLoc; // local copy of time, for ease of capture
  Problem *Alf = new Problem(qDev, confFile, &grid, DD, out);
  Alf->Alfven(v, u);  // Inits v and u in DEVICE, calls BCex, prints ICs.

  //-- SYCL ranges and related accessories
  int      gOff[]=      {grid.h[0], grid.h[1], grid.h[2]};
  range<3> rStd  = range(grid.n[0], grid.n[1], grid.n[2]);
  range<1> linear_range  = range(stop_idx - start_idx);
  field   *aMax  = malloc_shared<field>(3, qDev), vChar; // For reduction, and CFL in timestepping

  //-- Main Evolution loop
  Log->togglePcontrol(1); // start profiling
  while( (Alf->t() <= Alf->tMax()) && (Alf->iStep() < Alf->nStep()) ){

    for (int irk = 0; irk < NRK; irk++){  //-- RK loop
      if (!irk){ aMax[0]=0.0; aMax[1]=0.0; aMax[2]=0.0; }

#if SYCL==oneAPI
      for(unsigned myDir=0; myDir<3; myDir++){ //-- Direction loop
        qDev.parallel_for<class parForFluxes>(rStd,
             mysycl::reduction(aMax+myDir, my1api::maximum<field>()), // mysycl::property::reduction::initialize_to_identity()),
             [=](id<3> id, auto &max){
#else // OpenSYCL, LLVM and oneAPIold cases
      range<3> rLoc  = range(  locSize,   locSize,   locSize);
      for(unsigned myDir=0; myDir<3; myDir++){ //-- Direction loop
        auto maxReduction = mysycl::reduction(aMax + myDir, mysycl::maximum<field>());
        qDev.parallel_for(nd_range<3> (rStd, rLoc), maxReduction, [=](nd_item<3> it, auto &max) {
             id<3> id = it.get_global_id();           // All needed global indexes
#endif
            int myId = globLinId(id, grid.nh, gOff); // Accessing v, u and the like
            // What you declare here resides in GPU core-memory >~ 32 kB
            // BUT few registers, so declare things as you need them. - SC
            int dStride= stride(id, myDir, grid.nh); // Stride from the above indexes
            field vR[FLD_TOT], vL[FLD_TOT];
            for (int i=0; i < FLD_TOT; ++i){ holibRec(myId, v[i], dStride, vL+i,  vR+i); }

            Metric g(grid.xC(id, 0), grid.xC(id, 1), grid.xC(id, 2));
            field uR[FLD_TOT], fR[FLD_TOT], vfR[2], vtR[2];  physicalFlux(myDir, g, vR, uR, fR, vfR, vtR);
            field uL[FLD_TOT], fL[FLD_TOT], vfL[2], vtL[2];  physicalFlux(myDir, g, vL, uL, fL, vfL, vtL);

            field ap = mysycl::max((field)0., mysycl::max( vfL[0], vfR[0]));
            field am = mysycl::max((field)0., mysycl::max(-vfL[1],-vfR[1]));
#ifdef UCT  // For induction we save these too
            apG[myDir][myId] = ap;  vt1[myDir][myId] =(ap*vtL[0]+am*vtR[0])/(ap+am);
            amG[myDir][myId] = am;  vt2[myDir][myId] =(ap*vtL[1]+am*vtR[1])/(ap+am);
#endif
            // Fluxes from reconstructed values. When CT is on, this loop leaves B fields out
            for (int i=0; i<FLD_TOT; ++i){ f[i][myId] = (ap*fL[i]+am*fR[i]-ap*am*(uR[i]-uL[i]))/(ap+am); }

            // For timestepping; needed only if 0==irk
            if(!irk){field localMax = mysycl::max(ap, am);  max.combine(localMax); }

#ifndef NDEBUG  // Variables you may print for debug. set debug:= <whatYouWantToSee>
            for (int i=0; i<FLD_TOT; ++i){ debug[i][myId] = vL[i]; }
            // Some examples for what you may want to debug!
            debug[1][myId] = ap;    debug[2][myId] = am;    debug[3][myId] = ap+am;
            debug[4][myId] = uR[0]; debug[5][myId] =-uL[0]; debug[6][myId] = uR[0]-uL[0];
#endif
        }); // end parallel_for
        qDev.wait_and_throw(); // Now the flux is available everywhere

        //-- Flux reconstr & Derivatives. Trying to provide more halos to skip this was not beneficial!
        DD->BCex(myDir, grid, f); // call BCEX on fluxes.

#ifndef NDEBUG
        // Warning: Uncommenting this results in three full dumps per timestep (one per spatial direction)
	// Alf->dump(f); Alf->dump(debug);
#endif

        const field multiplier = field(1) / grid.dx[myDir];
        const int dStride = strides[myDir];
        for (int iVar=0; iVar<FLD_TOT; ++iVar){
          field *fi = f[iVar], *dui = du[iVar];
          qDev.parallel_for<class parForUpdateDu>(linear_range, [=](item<1> it) { // Update du with current direction
            int myId = start_idx+it[0];
            field d = (myDir ? dui[myId] : 0);
            d += multiplier * holibDer(myId, fi, dStride);
            dui[myId] = d;
          }).wait_and_throw();
        }

      } // - End loop on directions

      if (0 == irk){ // - Only at the end of 1st RK step compute the timestep (less MPI barriers)
#ifdef MPICODE
        MPI_Allreduce(MPI_IN_PLACE, aMax, 3, MPI_FIELD, MPI_MAX, MPI_COMM_WORLD ); // MPI_COMM_WORLD is an epsilon faster than DD->cartComm()
#endif
        vChar=std::max( {aMax[0]/grid.dx[0], aMax[1]/grid.dx[1], aMax[2]/grid.dx[2]} );  // Accumulation
        Alf->dtUpdate(vChar); dtLoc = Alf->dt(); // Update timing
        for (int i=0; i<FLD_TOT; ++i) { qDev.memcpy(u0[i], u[i], Ncell*sizeof(field)); } // Store original u: u0 = u
        qDev.wait_and_throw();
      }

      for (int iVar=0; iVar<FLD_TOT; ++iVar){
        field *dui = du[iVar], *u0i = u0[iVar], *ui = u[iVar];
        qDev.parallel_for<class parForRK>(linear_range, [=](item<1> it) { // Update u with du
          int myId = start_idx+it[0];
          ui[myId] = crk1[irk] * u0i[myId] + crk2[irk]*( ui[myId] - dtLoc*dui[myId] );
        });
      }
      qDev.parallel_for<class parForCons2Prim>(rStd, [=](item<3> it) { // Convert conserved variables to primitive variables
        auto id = it.get_id();
        int myId   = globLinId(id, grid.nh, grid.h ); // Accessing v, u and the like
        Metric g(grid.xC(id, 0), grid.xC(id, 1), grid.xC(id, 2));
        cons2prim(myId, Ncell, u, v, g);
      });
      qDev.wait_and_throw();

      for(unsigned myDir=0; myDir<3; myDir++) { DD->BCex(myDir, grid, v); }

    }//-- END RK

    // Print timestep report
    Log->setPar(true);
    *Log+4<<TAG<<"Step "<< Alf->iStep()<<" out "<< Alf->iOut()-1;
    *Log<<" characteristic: "<< vChar;
    *Log<<" in " << Alf->time()<<" s"; Log->fl();
#ifndef NDEBUG
    // Alf->dump(u);
#endif
    if( Alf->t() / Alf->tMax() > Alf->iOut() * Alf->tOut() ){
      Log->togglePcontrol(0); // When profiling, exclude output
      Alf->dump(v);
      Log->togglePcontrol(1);
    }
  } // Evolution while

  Log->togglePcontrol(0);

  for (int i=0; i < FLD_TOT; ++i){ free(  out[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(    v[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(    u[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(   u0[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(    f[i], qDev); }
  for (int i=0; i < FLD_TOT; ++i){ free(   du[i], qDev); }
#ifndef NDEBUG
  for (int i=0; i < FLD_TOT; ++i){ free(debug[i], qDev); }
#endif
#ifdef UCT
  for (int i=0; i < 3; ++i){ free(  apG[i], qDev); }
  for (int i=0; i < 3; ++i){ free(amGdu[i], qDev); }
  for (int i=0; i < 3; ++i){ free(  vt1[i], qDev); }
  for (int i=0; i < 3; ++i){ free(  vt2[i], qDev); }
#endif

  Log->setPar(false);
  Log->cleanup();
  return 0;

} // end main
