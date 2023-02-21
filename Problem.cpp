//  Copyright(C) 2021 Salvatore Cielo, LRZ
//  Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Problem.hpp"
#include "Metric.hpp"
#include <iomanip>

using namespace std;

Problem::Problem(mysycl::queue qx, string &confFile, Grid *grid, Domain *D, field_array &fld ){
  Log = Logger::getInstance(); grid_ = grid; D_ = D; N_ = grid_->nht;
  iOut_ = 0; iStep_ = 0; nStep_ = 0;  dumpHalos = false; locSize = 1;
  tMax_   = 1.0; dt_ = 0.0; t_ = 0.0, tOut_=0.025, cfl_ = 0.8/3.0; // Divide by 3 as it's 3D
  qq = qx;
  stepTime_.init();
  std::ifstream inFile(confFile);   std::string key, val;
  Log->setPar(false); *Log+3<<TAG<<"Reading input: ";
  while (std::getline(inFile, key, ' ') && std::getline(inFile, val)){
    if(!key.compare("tMax" )){ tMax_  = static_cast<field>(stod(val)); *Log<<"\n\ttMax  "<<tMax_  ; continue;}
    if(!key.compare("nStep")){ nStep_ = stoi(val); *Log<<"\n\tnStep "<<nStep_ ; continue;}
    if(!key.compare("tOut" )){ tOut_  = stof(val); *Log<<"\n\ttOut " <<tOut_  ; continue;}
    if(!key.compare("dumpHalos")){ dumpHalos= (bool)  stoi(val); *Log<<"\n\tdumpHalos " << dumpHalos ; continue;}
    if(!key.compare("locSize")){locSize=stoi(val); *Log<<"\n\tlocSize" << locSize; continue;}
  }Log->fl();
  nxNH_ = D_->cartDims(0) * grid_->n[0];
  nyNH_ = D_->cartDims(1) * grid_->n[1];
  nzNH_ = D_->cartDims(2) * grid_->n[2];
  if(NULL == fld ){ Log->Error("%s Allocate var and assign fld before initializing the problem.", TAG); return; }
  for(int iVar=0; iVar<FLD_TOT; ++iVar) out[iVar] = fld[iVar];
  Log->setPar(false); *Log+6<<TAG<<"Problem framework of size "<<N_<< " and output dir created."; Log->fl();
  // Option to chenge the order of BOV output for the various MPI ranks. E.g. a z-first approach is:
  // BOVRank_=D_->cartCoords(2)+( D_->cartCoords(1)+D_->cartCoords(0)*D_->cartDims(1) )*D_->cartDims(2);
  // One could think of coding the z-first/x-first option it in CMake.
  BOVRank_ = 0; // x-first
#ifdef MPICODE
  MPI_Comm_rank(D_->cartComm(), (int*)&BOVRank_);
#endif
}

//-- Timing and Output (this class has all, useless to make another one)
void Problem::dtUpdate(field aMax){
  dt_ = std::min(cfl_/aMax, tMax_ -t_ + 1.e-6*tMax_);  t_ += dt_; iStep_++;
  dtPrint();
}
void Problem::dtPrint( ){
  Log->setPar(false);
  *Log+0<<TAG <<" Step # "<< iStep_ <<": t "<< t_  <<", i.e. "<< t_/tMax_*100.0 <<"%, dt "<< dt_;   Log->fl();
}

void Problem::dump(field_array &v){ // Asynchronous output
#if defined(FILE_IO_VISIT_BOV)
  // Device code: update *out with provided field
  Grid gr=*grid_;
  if(dumpHalos){
    for(int iVar=0; iVar<FLD_TOT; ++iVar) qq.memcpy(out[iVar], v[iVar], gr.nht*sizeof(field)); // Direct memcpy
  } else {  // Manual indexing necessary
    field *vt[FLD_TOT]; for (int i = 0; i < FLD_TOT; i++) vt[i] = v[i];
    field *outt[FLD_TOT]; for (int i = 0; i < FLD_TOT; i++) outt[i] = out[i];
    qq.parallel_for<class parForDump>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
      auto iOut= it.get_linear_id(); // Output array has NH halo scope here
      auto iV  = globLinId(it.get_id(), gr.nh, gr.h); // v has WH indexing; offset by halos
      for(int iVar=0; iVar<FLD_TOT; ++iVar)
        outt[iVar][iOut] = v[iVar][iV];
    });
  }
  qq.wait_and_throw();
  // Host code: simple but parallel BOV output
  output::writeArray("out", "task", *this);
#elif defined(FILE_IO_DISABLED)
  // Do nothing.
#else
#warn "FILE_IO variable is not valid"
#endif
  iOut_++;
}

void Problem::InitConstWH(field *v, field val) { // HOST CODE: kernel for initialization.
  if(!v){ Log->Error("%s Array was not initialized.", TAG); return; }
  qq.parallel_for<class parForInitConstWH>(range<3>(grid_->nh[0], grid_->nh[1], grid_->nh[2]), [=, gr = *(this->grid_)](item<3> it) {
    int offset[3] = {0,0,0};
    auto iV  = globLinId(it, gr.nh, offset); // v has WH indexing; offset by halos
    v[iV] = val;
  });
}

void Problem::InitConstNH(field *v, field val) { // HOST CODE: kernel for initialization.
  if(!v){ Log->Error("%s Array was not initialized.", TAG); return; }
  qq.parallel_for<class parForInitConstNH>(range<3>(grid_->n[0], grid_->n[1], grid_->n[2]), [=, gr = *(this->grid_)](item<3> it) {
    auto iV  = globLinId(it, gr.nh, gr.h); // v has WH indexing; offset by halos
    v[iV] = val;
  });
}

////-- Problem-specific ICs
void Problem::TestUniform(field_array &v, field xx){ // HOST CODE: Initializing
  InitConstWH(v[RH], xx);  InitConstWH(v[PG], 1.); // this is all device code
  InitConstWH(v[VX], .5);  InitConstWH(v[VY], .5); InitConstWH(v[VZ], .5);
  InitConstWH(v[BX], 0.);  InitConstWH(v[BY], 0.); InitConstWH(v[BZ], 0.);
  qq.wait_and_throw();
  Log->setPar(false);  *Log+0<<TAG<<"Initialized Problem Uniform. Call prim2cons and cons2prim for phyisical correctness!"; Log->fl();
}

void Problem::Alfven(field_array &v, field_array &u){ // HOST CODE: Initializing
  field alfRH = 1.0, alfB0 = 1.0, alfPG = 1.0, alfAmp=1.0;
  int alfLx = 1, alfLy = 1, alfLz = 1;
  stepTime_.on();
  Log->setPar(false); *Log+3<<TAG<<"Reading input: ";
  ifstream inFile("echo.par");   string key, value;
  while (std::getline(inFile, key, ' ') && std::getline(inFile, value)){
    if(!key.compare("alfRH" )){ alfRH = stof(value); *Log<<"\n\talfRH  " << alfRH ; continue;}
    if(!key.compare("alfB0" )){ alfB0 = stof(value); *Log<<"\n\talfB0  " << alfB0 ; continue;}
    if(!key.compare("alfPG" )){ alfPG = stof(value); *Log<<"\n\talfPG  " << alfPG ; continue;}
    if(!key.compare("alfLx" )){ alfLx = stof(value); *Log<<"\n\talfLx  " << alfLx ; continue;}
    if(!key.compare("alfLy" )){ alfLy = stof(value); *Log<<"\n\talfLy  " << alfLy ; continue;}
    if(!key.compare("alfLz" )){ alfLz = stof(value); *Log<<"\n\talfLz  " << alfLz ; continue;}
    if(!key.compare("alfAmp")){ alfAmp= stof(value); *Log<<"\n\talfAmp " << alfAmp; continue;}
    if(!key.compare("tMax"  )){ tMax_ = static_cast<field>(stod(value)); *Log<<"\n\ttMax   " << tMax_  ; continue;}
  } Log->fl();

  field kx = alfLx ? 2*M_PI/alfLx:0.0,  ky = alfLy ? 2*M_PI/alfLy:0.0, kz = alfLz ? 2*M_PI/alfLz:0.0;
  *Log+4<<TAG<<"kxyz "<<kx<<" "<<ky<<" "<<kz; Log->fl();

#if PHYSICS==MHD
  field va = alfB0 / std::sqrt(alfRH);
#elif PHYSICS==GRMHD
  field wt  = alfRH + (GAMMA1)*alfPG + alfB0*alfB0*(1+alfAmp*alfAmp);
  field tmp = 2*alfAmp*alfB0*alfB0/wt;
  field va  = alfB0 / std::sqrt( wt* 0.5 *(1.+std::sqrt(1.-tmp*tmp) ) );
  field vmul= 1.0/std::sqrt(1.0 - (alfAmp*alfAmp*va*va));
#endif
  if(1.0 == tMax_ ){ tMax_ = 2*M_PI / (va * std::hypot(kx, ky, kz) ); } // C++17 :)
  *Log<<TAG<<"tMax  is set to "<< tMax_ ; Log->fl();
  field alp = std::atan2(ky,kx), bet = std::atan2(kz,kx), gam = std::atan2(kz, std::hypot(kx, ky));

  *Log+4<<TAG<<"alp bet gam va "<<alp<<" "<<bet<<" "<<gam<<" "<<va; Log->fl();

  field rot[9]={ std::cos(alp)*std::cos(gam),-std::sin(alp),-std::cos(alp)*std::sin(gam),
                 std::sin(alp)*std::cos(gam), std::cos(alp),-std::sin(alp)*std::sin(gam),
                               std::sin(gam), 0.           ,               std::cos(gam) };
  //-- Device code
  field bS[]={D_->boxSize(0), D_->boxSize(1), D_->boxSize(2)};
  Grid gr = *grid_; // For ease of lambda capture
  qq.parallel_for<class parForProblemAlfven>(range(gr.n[0], gr.n[1], gr.n[2]), [=, NN=N_](item<3> it) {
    field phi = 0.0, bx, by, bz, vx, vy, vz;
    auto i = globLinId(it, gr.nh, gr.h); // Addressing fld: WH indexing

    phi = alfLz * (gr.xC(it,2)/bS[2]+0.5) + // Cell centers use it here, i.e. NH indexing -> fine
          alfLy * (gr.xC(it,1)/bS[1]+0.5) +
          alfLx * (gr.xC(it,0)/bS[0]+0.5) ;
    phi*= 2.0*M_PI;
    bx = alfB0; by = alfB0 *alfAmp *mysycl::cos(phi); bz = alfB0 *alfAmp *mysycl::sin(phi);
    vx = 0.   ; vy =-va    *alfAmp *mysycl::cos(phi); vz =-va    *alfAmp *mysycl::sin(phi);

    // Initialization
    v[VX][i] = rot[0]*vx + rot[1]*vy + rot[2]*vz;
    v[VY][i] = rot[3]*vx + rot[4]*vy + rot[5]*vz;
    v[VZ][i] = rot[6]*vx + rot[7]*vy + rot[8]*vz;
#if PHYSICS==GRMHD
    v[VX][i] *= vmul;  v[VY][i] *= vmul;  v[VZ][i] *= vmul;
#endif
    v[BX][i] = rot[0]*bx + rot[1]*by + rot[2]*bz;
    v[BY][i] = rot[3]*bx + rot[4]*by + rot[5]*bz;
    v[BZ][i] = rot[6]*bx + rot[7]*by + rot[8]*bz;
    v[RH][i] = alfRH;
    v[PG][i] = alfPG;

    // Same as all other problems: prim2cons and cons2prim
    id<3> id = it.get_id();
    Metric g(gr.xC(id, 0), gr.xC(id, 1), gr.xC(id, 2));
    prim2cons(i, gr.nht, v, u, g);
    cons2prim(i, gr.nht, u, v, g);
  }).wait_and_throw();

  // BCex. We could save calls if we paid attention to directions, but this is cleaner! -SC
  D_->BCex(2,gr,v);  D_->BCex(1,gr,v);  D_->BCex(0,gr,v);
  dump(v); // Print ICs
  Log->setPar(false);  *Log+0<<TAG<<"Initialized Problem Alfven in "<<stepTime_.lap(); Log->fl();
}

// This is a stub at the moment. Coded as ECHO included a Blastwave problem
void Problem::BlastWave(field_array &v){ // HOST CODE: Initializing
  for (unsigned iVar = 0; iVar < FLD_TOT; iVar++)
    InitConstWH(v[iVar], (field)(iVar*Log->getMyRank()));
  qq.wait_and_throw();
  Log->setPar(false);  *Log+0<<TAG<<"Initialized Problem BlastWave. Call prim2cons and cons2prim for physical correctness!"; Log->fl();
}
