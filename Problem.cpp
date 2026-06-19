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
#include "Grid.hpp"
#include "Metric.hpp"
#include "Physics.hpp"
#include "Solver.hpp"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <ios>

using namespace std;
using namespace sycl;

Problem::Problem(sycl::queue qx, Parameters &parFile, Grid *grid, Domain *D, real_array &fld ): config(parFile) {
  grid_ = grid; D_ = D; N_ = grid_->nht;
  iOut_ = 0; iStep_ = 0; nStep_ = config.getOr("nStep", 0);  dumpHalos = static_cast<bool>(config.getOr("dumpHalos", 0)); locSize = config.getOr("locSize", 1); fileIO_ = static_cast<bool>(config.getOr("fileIO", 1));
  tMax_   = config.getOr<real>("tMax", 1.0); dt_ = 0.0; dt_prev_ = 0.0; t_ = 0.0; tOut_ = config.getOr("tOut", 0.025); cfl_ = 0.8 / static_cast<real>(NDIM); // Divide by NDIM dimensions
  qq = qx;
  stepTime_.init();

  nxNH_ = D_->cartDims(0) * grid_->n[0];
  nyNH_ = D_->cartDims(1) * grid_->n[1];
  nzNH_ = D_->cartDims(2) * grid_->n[2];
  Log::Assert(fld != NULL, "Allocate var and assign fld before initializing the problem.");
  for(int iVar=0; iVar<FLD_TOT; ++iVar) out[iVar] = fld[iVar];
  Log::cout(6) << TAG << "Problem framework of size " << N_ << " and output dir created." << Log::endl;
  // Option to chenge the order of BOV output for the various MPI ranks. E.g. a z-first approach is:
  // BOVRank_=D_->cartCoords(2)+( D_->cartCoords(1)+D_->cartCoords(0)*D_->cartDims(1) )*D_->cartDims(2);
  // One could think of coding the z-first/x-first option it in CMake.
  BOVRank_ = Log::mpiRank(); // x-first
  // TASKS  CELL TIME SPEC ABS
  Log::cerr(8)<<"TASKS\t CELLS\t TIME\t SPEC \t ABS"<<Log::endl;
}

//-- Timing and Output (this class has all, useless to make another one)
void Problem::dtUpdate(real aMax){
  dt_ = std::min(cfl_/aMax, tOut_*(iOut_+1)*tMax_ -t_ + 1.e-6*tOut_*(iOut_+1)*tMax_);
  dt_ = std::min(dt_,       tMax_                   -t_ + 1.e-6*tMax_                );

  if (dt_ < 0) {
    Log::cout(0) << TAG << "ERROR: negative timestep (dt = " << dt_ << "). Dumping and aborting." << Log::endl;
    dump(v_); abort();
  }
  if (dt_ == 0) {
    Log::cout(0) << TAG << "dt = 0: steady state reached. Dumping final output." << Log::endl;
    dump(v_); t_ = tMax_ + 1.0; return;
  }

  static constexpr real GROWTH_LIMIT = 2.0;
  if (iStep_ > 0) dt_ = std::min(dt_, dt_prev_ * GROWTH_LIMIT);
  dt_prev_ = dt_;

  t_ += dt_;
  //-- ACHTUNG!! Here and only here we are resetting the step timer!
  double wallT_ = stepTime_.lap(false, true, BOVRank_); stepTime_.init();
  
  // Print on the main out for advancement
  Log::cout(0)<<TAG<<" Step # "<<iStep_<<": t "<<t_<<" i.e. "<<(t_/tMax_ * 100.0)<<"% dt "<<dt_<<" walltime/s "<<wallT_<<Log::endl;
  // Print perf values on the err
  long double cells = grid_->nt, spec = 1.0*cells /wallT_;
  if(iStep_) {
 
    Log::cerr(0)<< std::defaultfloat << Log::mpiSize() <<" "<< round(cbrt(cells * Log::mpiSize())) <<" "
                 << std::scientific   << wallT_<<" " <<spec <<" "<< Log::mpiSize() * spec
#ifdef TB_ENERGY
                 << " " << ((Log::mpiRanksPerNode() * cells) / stepTime_.lastEnergyReading()) << std::defaultfloat 
#endif
                 << Log::endl;
  }
  iStep_++;
}

void Problem::waitOut() {
  if (out_pending) {
    MPI_Waitall(FLD_TOT, out_req, MPI_STATUSES_IGNORE);
    for (int i = 0; i < FLD_TOT; ++i) MPI_File_close(&out_fh[i]);
    out_pending = false;
  }
}

void Problem::writeBOV(Grid &gr, std::string dir, std::string name) {
  Log::Assert(out[0] != nullptr, "Array was not initialized.");

  std::filesystem::create_directories(dir);

  std::ostringstream stepDir;
  stepDir << dir << "/" << std::setw(4) << std::setfill('0') << iOut_;
  std::filesystem::create_directories(stepDir.str());

  int Ncell[NDIM];
  real brickSize  [NDIM];
  real brickOrigin[NDIM] = {D_->boxMin (0), D_->boxMin (1), D_->boxMin (2)};

  for(int ii = 0; ii < NDIM; ++ii){
    if(dumpHalos){
      Ncell[ii] = gr.nh[ii];
      brickOrigin[ii]*=(1.0*gr.nh[ii])/gr.n[ii];
    } else {
      Ncell[ii] = gr.n[ii];
    }
    brickSize[ii] = Ncell[ii] * D_->cartDims(ii) * gr.dx[ii];
  }

  for (int iVar = 0; iVar < FLD_TOT; ++iVar) {

    if (Log::isMaster()) {
      std::ostringstream bovName;
      bovName << dir << "/" << varLabel[iVar] << "_"
              << std::setw(4) << std::setfill('0') << iOut_ << ".bov";
      ofstream bov; bov.open(bovName.str(), ios_base::out);
      bov << "TIME: " << t() << "\n";
      bov << "DATA_FILE: " << std::setw(4) << std::setfill('0') << iOut_
          << "/" << name << "_" << varLabel[iVar] << "_%04d.dat\n";
      bov << "DATA_SIZE: " << Ncell[2] * D_->cartDims(2) << " "
          << Ncell[1] * D_->cartDims(1) << " "
          << Ncell[0] * D_->cartDims(0) << "\n";
      bov << "DATA_FORMAT: " << FIELD_FORMAT << "\n";
      bov << "VARIABLE: v\n";
      bov << "DATA_ENDIAN: LITTLE\n";
      bov << "CENTERING: zonal\n";
      bov << "BRICK_ORIGIN: " << brickOrigin[2] << " " << brickOrigin[1] << " " << brickOrigin[0] << "\n";
      bov << "BRICK_SIZE: "  << brickSize[2]  << " " << brickSize[1]  << " " << brickSize[0]  << "\n";
      bov << "DIVIDE_BRICK: false\n";
      bov << "DATA_BRICKLETS: " << Ncell[2] << " " << Ncell[1] << " " << Ncell[0] << "\n";
      bov << "DATA_COMPONENTS: 1\n";
      bov << std::flush; bov.close();
    }
  }

  Log::cout(0) << TAG << "Dumped " << dir << " (" << name << ") output #" << iOut_ << Log::endl;
}

void Problem::dump(real_array &v, Grid &gr, std::string dir, std::string name){
  if (fileIO_) {
    waitOut();
    // Device code: update *out with provided real
    if(dumpHalos){
      for(int iVar=0; iVar<FLD_TOT; ++iVar) qq.memcpy(out[iVar], v[iVar], gr.nht*sizeof(real)); // Direct memcpy
    } else {  // Manual indexing necessary
      real *outt[FLD_TOT]; for (int i = 0; i < FLD_TOT; i++) outt[i] = out[i];
      qq.parallel_for<class parForDump>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
        auto iOut= it.get_linear_id(); // Output array has NH halo scope here
        auto iV  = globLinId(it.get_id(), gr.nh, gr.h); // v has WH indexing; offset by halos
        for(int iVar=0; iVar<FLD_TOT; ++iVar)
          outt[iVar][iOut] = v[iVar][iV];
      });
    }
    qq.wait_and_throw();
    // Host code: .bov headers
    writeBOV(gr, dir, name);
    // MPI: start async .dat writes
    {
      int Ncell[NDIM], Ntot = 1;
      for(int ii = 0; ii < NDIM; ++ii){
        if(dumpHalos) Ncell[ii] = gr.nh[ii];
        else          Ncell[ii] = gr.n [ii];
        Ntot *= Ncell[ii];
      }
      for (int iVar = 0; iVar < FLD_TOT; ++iVar) {
        std::ostringstream datName;
        datName << dir << "/" << std::setw(4) << std::setfill('0') << iOut_
                << "/" << name << "_" << varLabel[iVar] << "_"
                << std::setw(4) << std::setfill('0') << BOVRank_ << ".dat";
        MPI_File_open(MPI_COMM_SELF, datName.str().c_str(),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_fh[iVar]);
        MPI_File_iwrite(out_fh[iVar], out[iVar], Ntot, MPI_REAL, &out_req[iVar]);
      }
      out_pending = true;
    }
  }
  iOut_++;
}

void Problem::InitConstWH(real *v, real val) { // HOST CODE: kernel for initialization.
  Log::Assert(v, "Array was not initialized.");
  qq.parallel_for<class parForInitConstWH>(range<3>(grid_->nh[0], grid_->nh[1], grid_->nh[2]), [=, gr = *(this->grid_)](item<3> it) {
    int offset[NDIM] = {0,0,0};
    auto iV  = globLinId(it, gr.nh, offset); // v has WH indexing; offset by halos
    v[iV] = val;
  });
}

void Problem::InitConstNH(real *v, real val) { // HOST CODE: kernel for initialization.
  Log::Assert(v, "Array was not initialized.");
  qq.parallel_for<class parForInitConstNH>(range<3>(grid_->n[0], grid_->n[1], grid_->n[2]), [=, gr = *(this->grid_)](item<3> it) {
    auto iV  = globLinId(it, gr.nh, gr.h); // v has WH indexing; offset by halos
    v[iV] = val;
  });
}

void Problem::init(real_array &v, real_array &u) {
  using namespace std::string_literals;
  string problemName = config.getOr("problem", "INVALID"s);
  if (problemName == "Uniform"s) {
    Uniform(v, u);
  } else if (problemName == "Alfven"s) {
    Alfven(v,u);
  } else if (problemName == "Blastwave"s) {
    BlastWave(v, u);
  } else if (problemName == "Gradient"s) {
    Gradient(v, u);
  } else {
    Log::cout(0) << TAG << "Invalid problem " << problemName << ". The problem needs to be specified. Exiting." << Log::endl;
    abort();
  }
  config.report();
  for (int i = 0; i < FLD_TOT; ++i) { v_[i] = v[i]; u_[i] = u[i]; }
}

////-- Problem-specific ICs
void Problem::Uniform(real_array &v, real_array &u){ // HOST CODE: Initializing
  auto xx = config.getOr("uniConst", 1.0);
  InitConstWH(v[RH], xx);  InitConstWH(v[PG], 1.); // this is all device code
  InitConstWH(v[VX], .5);  InitConstWH(v[VY], .5); InitConstWH(v[VZ], .5);
  InitConstWH(v[BX], 0.);  InitConstWH(v[BY], 0.); InitConstWH(v[BZ], 0.);
  qq.wait_and_throw();
  Log::cout(0) << TAG << "Initialized Problem Uniform." << Log::endl;

  // Same as all other problems: prim2cons and cons2prim to ensure physical correctness.
  Grid gr = *grid_; // For ease of lambda capture
  qq.parallel_for(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
    auto i = globLinId(it, gr.nh, gr.h); // Addressing fld: WH indexing
    id<3> id = it.get_id();
    Metric g(gr.xC(id, 0), gr.xC(id, 1), gr.xC(id, 2));
    prim2cons(i, gr.nht, v, u, g);
    cons2prim(i, gr.nht, u, v, g);
  }).wait_and_throw();

  // BCex. Leave all directions for debug purposes with dumpHalos on! -SC
  D_->BCex(2,gr,v);  D_->BCex(1,gr,v);  D_->BCex(0,gr,v);
  dump(v); // Print ICs
  Log::cout(0) << TAG << "Initialized Problem Uniform in " << stepTime_.lap() << Log::endl;
}

void Problem::Alfven(real_array &v, real_array &u){ // HOST CODE: Initializing
  real alfRH = config.getOr<real>("alfRH", 1.0), alfB0 = config.getOr<real>("alfB0", 1.0), alfPG = config.getOr<real>("alfPG", 1.0), alfAmp=config.getOr<real>("alfAmp", 1.0);
  real alfLx = config.getOr<real>("alfLx", 1.0), alfLy = config.getOr<real>("alfLy", 1.0), alfLz = config.getOr<real>("alfLz", 1.0);
  tMax_ = config.getOr<real>("tMax", 1.0);
  stepTime_.on();

  real kx = alfLx ? 2*M_PI/alfLx:0.0,  ky = alfLy ? 2*M_PI/alfLy:0.0, kz = alfLz ? 2*M_PI/alfLz:0.0;
  Log::cout(4) << TAG << "kxyz " << kx << " " << ky << " " << kz << Log::endl;

#if PHYSICS==MHD
  real va = alfB0 / std::sqrt(alfRH);
#elif PHYSICS==GRMHD
  real wt  = alfRH + (GAMMA1)*alfPG + alfB0*alfB0*(1+alfAmp*alfAmp);
  real tmp = 2*alfAmp*alfB0*alfB0/wt;
  real va  = alfB0 / std::sqrt( wt* 0.5 *(1.+std::sqrt(1.-tmp*tmp) ) );
  real vmul= 1.0/std::sqrt(1.0 - (alfAmp*alfAmp*va*va));
#endif
  if(1.0 == tMax_ ){ tMax_ = 2*M_PI / (va * std::hypot(kx, ky, kz) ); } // C++17 :)
  Log::cout(0) << TAG << "tMax  is set to " << tMax_ << Log::endl;
  real alp = std::atan2(ky,kx), bet = std::atan2(kz,kx), gam = std::atan2(kz, std::hypot(kx, ky));

  Log::cout(4) << TAG << "alp bet gam va " << alp << " " << bet << " " << gam << " " << va << Log::endl;

  real rot[9]={ std::cos(alp)*std::cos(gam),-std::sin(alp),-std::cos(alp)*std::sin(gam),
                 std::sin(alp)*std::cos(gam), std::cos(alp),-std::sin(alp)*std::sin(gam),
                               std::sin(gam), 0.           ,               std::cos(gam) };
  //-- Device code
  real bS[]={D_->boxSize(0), D_->boxSize(1), D_->boxSize(2)};
  Grid gr = *grid_; // For ease of lambda capture
  qq.parallel_for<class parForProblemAlfven>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
    real phi = 0.0, bx, by, bz, vx, vy, vz;
    auto i = globLinId(it, gr.nh, gr.h); // Addressing fld: WH indexing

    phi = alfLz * (gr.xC(it,2)/bS[2]+0.5) + // Cell centers use it here, i.e. NH indexing -> fine
          alfLy * (gr.xC(it,1)/bS[1]+0.5) +
          alfLx * (gr.xC(it,0)/bS[0]+0.5) ;
    phi*= 2.0*M_PI;
    bx = alfB0; by = alfB0 *alfAmp *sycl::cos(phi); bz = alfB0 *alfAmp *sycl::sin(phi);
    vx = 0.   ; vy =-va    *alfAmp *sycl::cos(phi); vz =-va    *alfAmp *sycl::sin(phi);

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

  // BCex. Leave all directions for debug purposes with dumpHalos on! -SC
  D_->BCex(2,gr,v);  D_->BCex(1,gr,v);  D_->BCex(0,gr,v);
  dump(v); // Print ICs
  Log::cout() << TAG << "Initialized Problem Alfven in "<<stepTime_.lap() << Log::endl;
}

void Problem::BlastWave(real_array &v, real_array &u){ // HOST CODE: Initializing
  real r0 = config.getOr("r0", 0.8);
  real b0 = 1.0 / sycl::sqrt(2.0), rh0 = 1e-4, pg0 = 5e-3, rh1 = 1e-2, pg1 = 1.0;
  real bS[]={D_->boxSize(0), D_->boxSize(1), D_->boxSize(2)};
  Grid gr = *this->grid_;

  Log::cout(0) << TAG << "WARNING: The Blastwave Problem is not fully tested for lack of BCs. May yield inconsistent results. " << Log::endl;

  qq.parallel_for<class parForProblemBlastwave>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
    auto i = globLinId(it, gr.nh, gr.h);
    real xC = gr.xC(it,0) / bS[0], yC = gr.xC(it,1) / bS[1], zC = gr.xC(it,2) / bS[2];
    real r = sycl::sqrt(xC * xC + yC * yC + zC * zC);
    real f = sycl::max(1.0 / pown(1.0 + (r / r0), 16), 1e-6);

    // Initialization
    v[VX][i] = 0.0;
    v[VY][i] = 0.0;
    v[VZ][i] = 0.0;
    v[BX][i] = b0;
    v[BY][i] = b0;
    v[BZ][i] = 0.0;
    v[RH][i] = rh0 + (rh1 - rh0) * f;
    v[PG][i] = pg0 + (pg1 - pg0) * f;

    Metric g(gr.xC(it, 0), gr.xC(it, 1), gr.xC(it, 2));
    prim2cons(i, gr.nht, v, u, g);
    cons2prim(i, gr.nht, u, v, g);
  });
  qq.wait_and_throw();
  // BCex. Leave all directions for debug purposes with dumpHalos on! -SC
  D_->BCex(2,gr,v);  D_->BCex(1,gr,v);  D_->BCex(0,gr,v);
  dump(v); // Print ICs
  Log::cout(0) << TAG << "Initialized Problem Blastwave in " << stepTime_.lap() << Log::endl;
}

void Problem::Gradient(real_array &v, real_array &u){ // HOST CODE: Initializing
  real gradRho0 = config.getOr<real>("gradRho0", 1.0);
  real gradRho1 = config.getOr<real>("gradRho1", 2.0);
  real gradP0   = config.getOr<real>("gradP0",   1.0);
  real bS[]={D_->boxSize(0), D_->boxSize(1), D_->boxSize(2)};
  real bMin0 = D_->boxMin(0);
  Grid gr = *grid_;
  qq.parallel_for<class parForProblemGradient>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
    auto i = globLinId(it, gr.nh, gr.h);
    real x = (gr.xC(it, 0) - bMin0) / bS[0];  // normalized 0..1
    real rho = gradRho0 + (gradRho1 - gradRho0) * x;
    real pg  = gradP0 * rho / gradRho0;  // isothermal: P/ρ constant
    v[VX][i] = 0.0;
    v[VY][i] = 0.0;
    v[VZ][i] = 0.0;
    v[BX][i] = 0.0;
    v[BY][i] = 0.0;
    v[BZ][i] = 0.0;
    v[RH][i] = rho;
    v[PG][i] = pg;
    id<3> id = it.get_id();
    Metric g(gr.xC(id, 0), gr.xC(id, 1), gr.xC(id, 2));
    prim2cons(i, gr.nht, v, u, g);
    cons2prim(i, gr.nht, u, v, g);
  }).wait_and_throw();
  D_->BCex(2,gr,v);  D_->BCex(1,gr,v);  D_->BCex(0,gr,v);
  dump(v);
  Log::cout(0) << TAG << "Initialized Problem Gradient in " << stepTime_.lap() << Log::endl;
}
