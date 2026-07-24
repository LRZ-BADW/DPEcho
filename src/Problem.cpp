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
#include "Solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <vector>

using namespace std;
using namespace sycl;

Problem::Problem(sycl::queue qx, Parameters &parFile, Grid *grid, Domain *D, Physics *phys, real *fld, std::string runName): config(parFile), phys_(phys) {
  runName_ = config.getOr("runName", runName);
  grid_ = grid; D_ = D; N_ = grid_->nht;
  nFields_ = phys->fldTot();
  iOut_ = 0; iStep_ = 0; nStep_ = config.getOr("nStep", 0);  dumpHalos = static_cast<bool>(config.getOr("dumpHalos", 0)); locSize = config.getOr("locSize", 1); fileIO_ = static_cast<bool>(config.getOr("fileIO", 1));
  tMax_   = config.getOr<real>("tMax", 1.0); dt_ = 0.0; dt_prev_ = 0.0; t_ = 0.0; tOut_ = config.getOr("tOut", 0.025); cfl_ = 0.8 / static_cast<real>(NDIM); // Divide by NDIM dimensions
  v_ = new real*[nFields_]; u_ = new real*[nFields_];
  qq = qx;
  stepTime_.init();

  nxNH_ = D_->cartDims(0) * grid_->n[0];
  nyNH_ = D_->cartDims(1) * grid_->n[1];
  nzNH_ = D_->cartDims(2) * grid_->n[2];
  Log::Assert(fld != NULL, "Allocate var and assign fld before initializing the problem.");
  out = fld;
  Log::cout(6) << TAG << "Problem framework of size " << N_ << " and output dir created." << Log::endl;
  // Rank ordering: x-first by default. A z-first approach would be:
  // myRank_=D_->cartCoords(2)+( D_->cartCoords(1)+D_->cartCoords(0)*D_->cartDims(1) )*D_->cartDims(2);
  // One could think of coding the option in CMake.
  myRank_ = Log::mpiRank(); // x-first
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
  //-- WARNING: Here and only here we are resetting the step timer!
  double wallT_ = stepTime_.lap(false, true, myRank_); stepTime_.init();
  
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

void Problem::writeVTKAsync(Grid &gr, std::string dir, std::string name) {
  Log::Assert(out != nullptr, "Array was not initialized.");
  if (dumpHalos) return;

  int Ncell[NDIM];
  for(int ii = 0; ii < NDIM; ++ii)
    Ncell[ii] = gr.n[ii];
  int Ntot = Ncell[0] * Ncell[1] * Ncell[2];

  int Gx = nxNH_, Gy = nyNH_, Gz = nzNH_;
  int Lx = gr.n[0], Ly = gr.n[1], Lz = gr.n[2];
  int xs = D_->cartCoords(0) * Lx;
  int ys = D_->cartCoords(1) * Ly;
  int zs = D_->cartCoords(2) * Lz;
  int xe = xs + Lx;
  int ye = ys + Ly;
  int ze = zs + Lz;

  // Per-variable binary blocks: each = [uint32_t blockSize][Ntot*sizeof(real) data]
  size_t perVarData = static_cast<size_t>(Ntot) * sizeof(real);
  uint32_t perVarBlock = static_cast<uint32_t>(perVarData);
  size_t varBlkBytes = sizeof(uint32_t) + perVarData;

#ifdef SINGLE_PRECISION
  constexpr const char* VTK_REAL = "Float32";
#else
  constexpr const char* VTK_REAL = "Float64";
#endif

  std::filesystem::create_directories(dir);
  std::ostringstream stepDir;
  stepDir << dir << "/" << std::setw(4) << std::setfill('0') << iOut_;
  std::filesystem::create_directories(stepDir.str());

  // XML header with per-variable named DataArrays
  std::ostringstream hdr;
  hdr << "<?xml version=\"1.0\"?>\n";
  hdr << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
  hdr << "  <ImageData WholeExtent=\"0 " << Gx << " 0 " << Gy << " 0 " << Gz << "\""
      << " Origin=\"" << D_->boxMin(0) << " " << D_->boxMin(1) << " " << D_->boxMin(2) << "\""
      << " Spacing=\"" << gr.dx[0] << " " << gr.dx[1] << " " << gr.dx[2] << "\">\n";
  hdr << "    <Piece Extent=\"" << xs << " " << xe << " " << ys << " " << ye << " " << zs << " " << ze << "\">\n";
  hdr << "      <CellData>\n";
  for (unsigned v = 0; v < nFields_; v++) {
    size_t off = static_cast<size_t>(v) * varBlkBytes;
    hdr << "        <DataArray type=\"" << VTK_REAL << "\" Name=\"" << varLabel(v, phys_->isMagnetic())
        << "\" format=\"appended\" offset=\"" << off << "\"/>\n";
  }
  hdr << "      </CellData>\n";
  hdr << "      <FieldData>\n";
  hdr << std::setprecision(16) << std::scientific;
  hdr << "        <DataArray type=\"" << VTK_REAL << "\" Name=\"t\" format=\"ascii\" numberOfTuples=\"1\">" << t_ << "</DataArray>\n";
  hdr << "        <DataArray type=\"" << VTK_REAL << "\" Name=\"dt\" format=\"ascii\" numberOfTuples=\"1\">" << dt_ << "</DataArray>\n";
  hdr << std::defaultfloat;
  hdr << "        <DataArray type=\"UInt32\" Name=\"iStep\" format=\"ascii\" numberOfTuples=\"1\">" << iStep_ << "</DataArray>\n";
  hdr << "        <DataArray type=\"UInt32\" Name=\"iOut\" format=\"ascii\" numberOfTuples=\"1\">" << iOut_ << "</DataArray>\n";
  hdr << "      </FieldData>\n";
  hdr << "    </Piece>\n";
  hdr << "  </ImageData>\n";
  hdr << "  <AppendedData encoding=\"raw\">\n";
  hdr << "    _";
  std::string header = hdr.str();

  // Pack interleaved out[] into per-variable contiguous blocks
  size_t binSize = static_cast<size_t>(nFields_) * varBlkBytes;
  binBuf_ = new char[binSize];
  for (unsigned v = 0; v < nFields_; v++) {
    size_t blkOff = static_cast<size_t>(v) * varBlkBytes;
    *reinterpret_cast<uint32_t*>(binBuf_ + blkOff) = perVarBlock;
    real *varDst = reinterpret_cast<real*>(binBuf_ + blkOff + sizeof(uint32_t));
    for (int i = 0; i < Ntot; i++)
      varDst[i] = out[i * nFields_ + v];
  }

  // Open .vti, write header sync, start async write of packed binary
  std::ostringstream vtiName;
  vtiName << stepDir.str() << "/rank_"
          << std::setw(4) << std::setfill('0') << myRank_ << ".vti";
  MPI_File_open(MPI_COMM_SELF, vtiName.str().c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFh);
  MPI_File_write(outFh, header.data(), header.size(), MPI_BYTE, MPI_STATUS_IGNORE);
  MPI_File_iwrite(outFh, binBuf_, binSize, MPI_BYTE, &outReq);
  outPending = true;

  Log::cout(0) << TAG << "Dumped " << dir << " (" << name << ") output #" << iOut_ << Log::endl;
}

void Problem::writePVTI(std::string dir, std::string name, unsigned long out) {
  int Lx = grid_->n[0], Ly = grid_->n[1], Lz = grid_->n[2];
  int Gx = nxNH_, Gy = nyNH_, Gz = nzNH_;
  std::string dumpDir = std::filesystem::path(dir).filename().string();
  std::ostringstream stepDir;
  stepDir << std::setw(4) << std::setfill('0') << out;

#ifdef SINGLE_PRECISION
  constexpr const char* VTK_REAL = "Float32";
#else
  constexpr const char* VTK_REAL = "Float64";
#endif

  // .pvti goes one level above dir (e.g. runName/, not runName/dump/)
  std::ostringstream pvtiName;
  pvtiName << std::filesystem::path(dir).parent_path().string() << "/" << name << "_"
           << std::setw(4) << std::setfill('0') << out << ".pvti";
  ofstream pvti(pvtiName.str(), ios_base::out);

  pvti << "<?xml version=\"1.0\"?>\n";
  pvti << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
  pvti << "  <PImageData WholeExtent=\"0 " << Gx << " 0 " << Gy << " 0 " << Gz << "\""
       << " Origin=\"" << D_->boxMin(0) << " " << D_->boxMin(1) << " " << D_->boxMin(2) << "\""
       << " Spacing=\"" << grid_->dx[0] << " " << grid_->dx[1] << " " << grid_->dx[2] << "\">\n";
  pvti << "    <PCellData>\n";
  for (unsigned v = 0; v < nFields_; v++)
    pvti << "      <PDataArray type=\"" << VTK_REAL << "\" Name=\"" << varLabel(v, phys_->isMagnetic()) << "\"/>\n";
  pvti << "    </PCellData>\n";

  int Rx = D_->cartDims(0), Ry = D_->cartDims(1), Rz = D_->cartDims(2);
  for (int iz = 0; iz < Rz; iz++) {
    for (int iy = 0; iy < Ry; iy++) {
      for (int ix = 0; ix < Rx; ix++) {
        int coords[3] = {ix, iy, iz};
        int rank;
        MPI_Cart_rank(D_->cartComm(), coords, &rank);
        int ps = ix * Lx, pe = ps + Lx;
        int qs = iy * Ly, qe = qs + Ly;
        int rs = iz * Lz, re = rs + Lz;
        pvti << "    <Piece Extent=\"" << ps << " " << pe << " " << qs << " " << qe << " " << rs << " " << re << "\""
             << " Source=\"" << dumpDir << "/" << stepDir.str() << "/rank_" << std::setw(4) << std::setfill('0') << rank << ".vti\"/>\n";
      }
    }
  }
  pvti << "  </PImageData>\n";
  pvti << "</VTKFile>\n";
  pvti.close();
}

void Problem::waitOut() {
  if (outPending) {
    MPI_Wait(&outReq, MPI_STATUS_IGNORE);
    delete[] binBuf_;
    binBuf_ = nullptr;
    std::string footer = "\n  </AppendedData>\n</VTKFile>\n";
    MPI_File_write(outFh, footer.data(), footer.size(), MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_close(&outFh);
    outPending = false;
    // Master writes .pvti for the just-completed dump (in dir/, not dir/XXXX/)
    if (Log::isMaster() && prevValid_) {
      writePVTI(prevDir_, prevName_, prevOut_);
    }
  }
}

void Problem::dump(real_array &v, Grid &gr, std::string dir, std::string name){
  if (fileIO_) {
    waitOut();
    // Device code: interleave all variables into single out[] buffer
    real *out_ = out;
    int nf = nFields_;
    if(dumpHalos){
      qq.parallel_for<class parForDumpWH>(range(gr.nht), [=](id<1> i) {
        for(int iVar=0; iVar<nf; ++iVar)
          out_[i * nf + iVar] = v[iVar][i];
      });
    } else {
      qq.parallel_for<class parForDumpNH>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
        auto id = it.get_id();
        auto iOut = id[0] + id[1] * gr.n[0] + id[2] * gr.n[0] * gr.n[1];
        auto iV   = globLinId(id, gr.nh, gr.h);
        for(int iVar=0; iVar<nf; ++iVar)
          out_[iOut * nf + iVar] = v[iVar][iV];
      });
    }
    qq.wait_and_throw();
    // Host code: async .vti write (header sync, binary async)
    writeVTKAsync(gr, dir, name);
    // Stash for next waitOut()'s .pvti
    prevDir_ = dir; prevName_ = name; prevOut_ = iOut_; prevValid_ = true;
  }
  iOut_++;
}

void Problem::restart(real_array &v, real_array &u, std::string restartDir) {
  int restartStep = config.getOr("restartStep", -1);
  Log::Assert(restartStep >= 0, "restartStep must be >= 0 for restart");
  Log::Assert(out != nullptr, "Output array was not initialized.");

  int Lx = grid_->n[0], Ly = grid_->n[1], Lz = grid_->n[2];
  int Ntot = Lx * Ly * Lz;
  size_t perVarData = static_cast<size_t>(Ntot) * sizeof(real);
  size_t varBlkBytes = sizeof(uint32_t) + perVarData;

  std::ostringstream fpath;
  fpath << restartDir << "/dump/" << std::setw(4) << std::setfill('0') << restartStep
        << "/rank_" << std::setw(4) << std::setfill('0') << myRank_ << ".vti";
  std::string fname = fpath.str();

  std::ifstream f(fname, std::ios::binary);
  Log::Assert(f.good(), "restart: cannot open " + fname);
  std::vector<char> content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  f.close();
  std::string s(content.data(), content.size());

  // Parse inline FieldData from XML header
  auto fieldVal = [&](const char* name) -> std::string {
    std::string pat = "Name=\"" + std::string(name) + "\" format=\"ascii\" numberOfTuples=\"1\">";
    auto p = s.find(pat);
    if (p == std::string::npos) return "";
    p += pat.size();
    auto e = s.find("</DataArray>", p);
    return (e == std::string::npos) ? "" : s.substr(p, e - p);
  };

  std::string tStr = fieldVal("t"), dtStr = fieldVal("dt");
  std::string iStepStr = fieldVal("iStep"), iOutStr = fieldVal("iOut");
  Log::Assert(!tStr.empty() && !iStepStr.empty(), "restart: missing FieldData metadata in " + fname);

  t_     = std::stod(tStr);
  dt_    = dtStr.empty() ? 0.0 : std::stod(dtStr);
  dt_prev_ = dt_; // Previous timestep for growth limiter
  iStep_ = std::stoul(iStepStr);
  iOut_  = iOutStr.empty() ? 0UL : std::stoul(iOutStr) + 1; // Next dump after the one we loaded
  tolCons2Prim_ = 1.e-12; // Tighter tolerance after restart to reduce roundtrip error

  for (unsigned i = 0; i < nFields_; ++i) { v_[i] = v[i]; u_[i] = u[i]; }

  // Find appended binary start
  auto atag = s.find("<AppendedData encoding=\"raw\">");
  Log::Assert(atag != std::string::npos, "restart: missing AppendedData tag");
  auto us = s.find('_', atag);
  Log::Assert(us != std::string::npos, "restart: missing binary marker '_'");
  const char* binStart = content.data() + us + 1;

  // Host-side de-interleave: VTK order (x fastest, NH) -> WH-indexed device
  real* vtbuf = new real[Ntot];
  for (unsigned vi = 0; vi < nFields_; vi++) {
    const char* varBlk = binStart + vi * varBlkBytes;
    std::memcpy(vtbuf, varBlk + sizeof(uint32_t), perVarData);
    real* vDev = v[vi];
    qq.parallel_for(range(Lx, Ly, Lz), [=, gr = *(this->grid_)](item<3> it) {
      auto id = it.get_id();
      int iVtk = id[0] + id[1] * Lx + id[2] * Lx * Ly;
      auto iV = globLinId(id, gr.nh, gr.h);
      vDev[iV] = vtbuf[iVtk];
    });
    qq.wait_and_throw();
  }
  delete[] vtbuf;

  // Compute conserved variables from restored primitives
  {
    real c2pTol = tolCons2Prim_;
    Physics *p = phys_;
    qq.parallel_for(range(Lx, Ly, Lz), [=, gr = *(this->grid_)](item<3> it) {
      auto i = globLinId(it, gr.nh, gr.h);
      id<3> id = it.get_id();
      Metric g(gr.xC(id, 0), gr.xC(id, 1), gr.xC(id, 2));
      p->prim2cons(i, gr.nht, v, u, g);
      p->cons2prim(i, gr.nht, u, v, g, c2pTol);
    }).wait_and_throw();
  }

  D_->BCex(2, *grid_, v);
  D_->BCex(1, *grid_, v);
  D_->BCex(0, *grid_, v);

  outPending = false;
  prevValid_ = false;

  Log::cout(0) << TAG << "Restarted from dump " << restartStep
               << " (t=" << t_ << ", dt=" << dt_ << ", iStep=" << iStep_ << ", iOut=" << iOut_ << ")" << Log::endl;
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
  } else if (phys_->isMagnetic() && problemName == "Alfven"s) {
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
  for (unsigned i = 0; i < nFields_; ++i) { v_[i] = v[i]; u_[i] = u[i]; }
}

////-- Problem-specific ICs
void Problem::Uniform(real_array &v, real_array &u){ // HOST CODE: Initializing
  auto xx = config.getOr("uniConst", 1.0);
  InitConstWH(v[RH], xx);  InitConstWH(v[PG], 1.); // this is all device code
  InitConstWH(v[VX], .5);  InitConstWH(v[VY], .5); InitConstWH(v[VZ], .5);
  if (phys_->isMagnetic()) {
    InitConstWH(v[BX], 0.);  InitConstWH(v[BY], 0.); InitConstWH(v[BZ], 0.);
  }
  qq.wait_and_throw();
  Log::cout(0) << TAG << "Initialized Problem Uniform." << Log::endl;

  // Same as all other problems: prim2cons and cons2prim to ensure physical correctness.
  Grid gr = *grid_; // For ease of lambda capture
  Physics *p = phys_;
  qq.parallel_for(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
    auto i = globLinId(it, gr.nh, gr.h); // Addressing fld: WH indexing
    id<3> id = it.get_id();
    Metric g(gr.xC(id, 0), gr.xC(id, 1), gr.xC(id, 2));
    p->prim2cons(i, gr.nht, v, u, g);
    p->cons2prim(i, gr.nht, u, v, g);
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
  bool isMag = phys_->isMagnetic();
  int physType = phys_->type();

  real kx = alfLx ? 2*M_PI/alfLx:0.0,  ky = alfLy ? 2*M_PI/alfLy:0.0, kz = alfLz ? 2*M_PI/alfLz:0.0;
  Log::cout(4) << TAG << "kxyz " << kx << " " << ky << " " << kz << Log::endl;

  real va;
  if (phys_->type() == Physics::MHD) {
    va = alfB0 / std::sqrt(alfRH);
  } else {
    real wt  = alfRH + (GAMMA1)*alfPG + alfB0*alfB0*(1+alfAmp*alfAmp);
    real tmp = 2*alfAmp*alfB0*alfB0/wt;
    va  = alfB0 / std::sqrt( wt* 0.5 *(1.+std::sqrt(1.-tmp*tmp) ) );
  }
  real vmul = (phys_->type() == Physics::GRMHD) ? 1.0/std::sqrt(1.0 - (alfAmp*alfAmp*va*va)) : 1.0;
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
  Physics *p = phys_;
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
    if (physType == Physics::GRMHD) {
      v[VX][i] *= vmul;  v[VY][i] *= vmul;  v[VZ][i] *= vmul;
    }
    if (isMag) {
      v[BX][i] = rot[0]*bx + rot[1]*by + rot[2]*bz;
      v[BY][i] = rot[3]*bx + rot[4]*by + rot[5]*bz;
      v[BZ][i] = rot[6]*bx + rot[7]*by + rot[8]*bz;
    }
    v[RH][i] = alfRH;
    v[PG][i] = alfPG;

    // Same as all other problems: prim2cons and cons2prim
    id<3> id = it.get_id();
    Metric g(gr.xC(id, 0), gr.xC(id, 1), gr.xC(id, 2));
    p->prim2cons(i, gr.nht, v, u, g);
    p->cons2prim(i, gr.nht, u, v, g);
  }).wait_and_throw();

  // BCex. Leave all directions for debug purposes with dumpHalos on! -SC
  D_->BCex(2,gr,v);  D_->BCex(1,gr,v);  D_->BCex(0,gr,v);
  dump(v); // Print ICs
  Log::cout() << TAG << "Initialized Problem Alfven in "<<stepTime_.lap() << Log::endl;
}

void Problem::BlastWave(real_array &v, real_array &u){ // HOST CODE: Initializing
  real r0 = config.getOr("r0", 0.8);
  real rh0 = 1e-4, pg0 = 5e-3, rh1 = 1e-2, pg1 = 1.0;
  bool isMag = phys_->isMagnetic();
  real b0 = isMag ? 1.0 / sycl::sqrt(2.0) : 0.0;
  real bS[]={D_->boxSize(0), D_->boxSize(1), D_->boxSize(2)};
  Grid gr = *this->grid_;

  Log::cout(0) << TAG << "WARNING: The Blastwave Problem is not fully tested for lack of BCs. May yield inconsistent results. " << Log::endl;

  Physics *p = phys_;
  qq.parallel_for<class parForProblemBlastwave>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
    auto i = globLinId(it, gr.nh, gr.h);
    real xC = gr.xC(it,0) / bS[0], yC = gr.xC(it,1) / bS[1], zC = gr.xC(it,2) / bS[2];
    real r = sycl::sqrt(xC * xC + yC * yC + zC * zC);
    real f = sycl::max(1.0 / pown(1.0 + (r / r0), 16), 1e-6);

    // Initialization
    v[VX][i] = 0.0;
    v[VY][i] = 0.0;
    v[VZ][i] = 0.0;
    if (isMag) {
      v[BX][i] = b0;
      v[BY][i] = b0;
      v[BZ][i] = 0.0;
    }
    v[RH][i] = rh0 + (rh1 - rh0) * f;
    v[PG][i] = pg0 + (pg1 - pg0) * f;

    Metric g(gr.xC(it, 0), gr.xC(it, 1), gr.xC(it, 2));
    p->prim2cons(i, gr.nht, v, u, g);
    p->cons2prim(i, gr.nht, u, v, g);
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
  bool isMag = phys_->isMagnetic();
  real bS[]={D_->boxSize(0), D_->boxSize(1), D_->boxSize(2)};
  real bMin0 = D_->boxMin(0);
  Grid gr = *grid_;
  Physics *p = phys_;
  qq.parallel_for<class parForProblemGradient>(range(gr.n[0], gr.n[1], gr.n[2]), [=](item<3> it) {
    auto i = globLinId(it, gr.nh, gr.h);
    real x = (gr.xC(it, 0) - bMin0) / bS[0];  // normalized 0..1
    real rho = gradRho0 + (gradRho1 - gradRho0) * x;
    real pg  = gradP0 * rho / gradRho0;  // isothermal: P/ρ constant
    v[VX][i] = 0.0;
    v[VY][i] = 0.0;
    v[VZ][i] = 0.0;
    if (isMag) {
      v[BX][i] = 0.0;
      v[BY][i] = 0.0;
      v[BZ][i] = 0.0;
    }
    v[RH][i] = rho;
    v[PG][i] = pg;
    id<3> id = it.get_id();
    Metric g(gr.xC(id, 0), gr.xC(id, 1), gr.xC(id, 2));
    p->prim2cons(i, gr.nht, v, u, g);
    p->cons2prim(i, gr.nht, u, v, g);
  }).wait_and_throw();
  D_->BCex(2,gr,v);  D_->BCex(1,gr,v);  D_->BCex(0,gr,v);
  dump(v);
  Log::cout(0) << TAG << "Initialized Problem Gradient in " << stepTime_.lap() << Log::endl;
}
