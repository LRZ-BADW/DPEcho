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
#include "utils/tb-types.hpp"
#include "utils/tb-timer.hpp"

#include <filesystem>

class Problem {

  public:
    int locSize;
    bool fileIO_, dumpHalos;
    real *out[FLD_TOT]; // Just to print

    Problem(sycl::queue q, Parameters &param, Grid *g, Domain *f, real_array &out);
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
    inline unsigned long BOVRank(){return BOVRank_;}
    // Output
    void dtUpdate(real);
    void dump( real_array &fld, Grid &gr, std::string dir="out", std::string name="task");
    void dump( real_array &fld, std::string dir="out", std::string name="task"){ dump(fld,*(this->grid_),dir,name ); };
    std::string getTimings() { return stepTime_.getTimings(); }
    void waitOut();

    // Generic problem initializer -- calls specific inits based on config
    void init(real_array &v, real_array &u);

    // Specific problems
    void Uniform  (real_array &v, real_array &u);
    void Alfven   (real_array &v, real_array &u);
    void BlastWave(real_array &v, real_array &u);

  private:
    void writeBOV(Grid &gr, std::string dir, std::string name);
    static constexpr const char* varLabel[FLD_TOT] = {
        "RH", "VX", "VY", "VZ", "PG", "BX", "BY", "BZ"
    };
    Parameters &config;
    sycl::queue qq;
    TB::Timer stepTime_;
    real tMax_, t_, dt_, cfl_, tOut_;
    real *v_[FLD_TOT], *u_[FLD_TOT], dt_prev_;
    unsigned int N_, nxNH_, nyNH_, nzNH_;
    unsigned long BOVRank_; // Necessary as BOV output assumes zyx output order
    unsigned long iOut_, iStep_, nStep_;
    Grid   *grid_;
    Domain *D_;
    MPI_File out_fh[FLD_TOT];
    MPI_Request out_req[FLD_TOT];
    bool out_pending = false;
};

#endif
