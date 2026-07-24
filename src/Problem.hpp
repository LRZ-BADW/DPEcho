//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Problem_hpp_
#define _Problem_hpp_

#include "echo.hpp"
#include "Domain.hpp"
#include "Grid.hpp"
#include "Parameters.hpp"
#include "Physics.hpp"
#include "utils/tb-types.hpp"
#include "utils/tb-timer.hpp"

#include <filesystem>

class Problem {

  public:
    int locSize;
    bool fileIO_, dumpHalos;
    real *out; // Interleaved output buffer (Ncell * nFields_)

    Problem(sycl::queue q, Parameters &param, Grid *g, Domain *f, Physics *phys, real *out, std::string runName = "task");
    ~Problem() { delete[] v_; delete[] u_; }
    void InitRampWH (real *);
    void InitRampNH (real *);
    void InitConstWH(real *, real );
    void InitConstNH(real *, real );
    inline real tMax (){return  tMax_ ;}
    inline real tOut (){return  tOut_ ;}
    inline real t    (){return  t_    ;}
    inline real dt   (){return  dt_   ;}
    inline real cfl  (){return  cfl_  ;}
    inline real lap  (bool keep=true){return  stepTime_.lap(keep);}
    inline unsigned long iOut   (){return iOut_   ;}
    inline unsigned long iStep  (){return iStep_  ;}
    inline unsigned long nStep  (){return nStep_  ;}
    inline unsigned long myRank(){return myRank_;}
    inline real tolCons2Prim() const { return tolCons2Prim_; }
    // Output
    void dtUpdate(real);
    void dump( real_array &fld, Grid &gr, std::string dir, std::string name);
    void dump( real_array &fld, std::string dir="", std::string name="") {
      std::string d = dir.empty() ? runName_ + "/dump" : runName_ + "/" + dir;
      std::string n = name.empty() ? runName_ : name;
      dump(fld, *(this->grid_), d, n);
    };
    void waitOut();
    std::string getTimings() { return stepTime_.getTimings(); }

    // Restart from a previous VTK dump
    void restart(real_array &v, real_array &u, std::string restartDir);

    // Generic problem initializer -- calls specific inits based on config
    void init(real_array &v, real_array &u);

    // Specific problems
    void Uniform  (real_array &v, real_array &u);
    void Alfven   (real_array &v, real_array &u);
    void BlastWave(real_array &v, real_array &u);
    void Gradient (real_array &v, real_array &u);

  private:
    void writeVTKAsync(Grid &gr, std::string dir, std::string name);
    void writePVTI(std::string dir, std::string name, unsigned long out);
    static const char* varLabel(int i, bool magnetic) {
      static const char* mag[]  = {"RH", "VX", "VY", "VZ", "PG", "BX", "BY", "BZ"};
      static const char* hydro[] = {"RH", "VX", "VY", "VZ", "PG"};
      return magnetic ? mag[i] : hydro[i];
    }
    std::string runName_;
    Parameters &config;
    sycl::queue qq;
    Physics *phys_;
    TB::Timer stepTime_;
    real tMax_, t_, dt_, cfl_, tOut_;
    real **v_, **u_;
    real dt_prev_, tolCons2Prim_ = 1.e-9;
    unsigned int N_, nFields_, nxNH_, nyNH_, nzNH_;
    unsigned long myRank_;
    unsigned long iOut_, iStep_, nStep_;
    Grid   *grid_;
    Domain *D_;
    // Async MPI I/O state
    bool     outPending = false;
    MPI_File   outFh;
    MPI_Request outReq;
    char      *binBuf_ = nullptr;
    // Stash for one-dump-behind .pvti
    std::string prevDir_, prevName_;
    unsigned long prevOut_ = 0;
    bool       prevValid_ = false;
};

#endif
