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
#include "Output.hpp"
#include "Parameters.hpp"
#include "utils/tb-types.hpp"
#include "utils/tb-timer.hpp"

class Problem {

  friend void output::writeArray(Problem &, Grid &, std::string, std::string);

  public:
    int locSize;
    bool dumpHalos;
    field *out[FLD_TOT]; // Just to print

    Problem(sycl::queue q, Parameters &param, Grid *g, Domain *f, field_array &out);
    void InitRampWH (field *);
    void InitRampNH (field *);
    void InitConstWH(field *, field );
    void InitConstNH(field *, field );
    inline field tMax (){return  tMax_ ;}
    inline field tOut (){return  tOut_ ;}
    inline field t    (){return  t_    ;}
    inline field dt   (){return  dt_   ;}
    inline field cfl  (){return  cfl_  ;}
    inline field lap  (bool keep=true){return  stepTime_.lap(keep);}
    inline unsigned long iOut   (){return iOut_   ;}
    inline unsigned long iStep  (){return iStep_  ;}
    inline unsigned long nStep  (){return nStep_  ;}
    inline unsigned long BOVRank(){return BOVRank_;}
    // Output
    void dtUpdate(field);
    void dump( field_array &fld, Grid &gr, std::string dir="out", std::string name="task");
    void dump( field_array &fld, std::string dir="out", std::string name="task"){ dump(fld,*(this->grid_),dir,name ); };
    std::string getTimings() { return stepTime_.getTimings(); }

    // Generic problem initializer -- calls specific inits based on config
    void init(field_array &v, field_array &u);

    // Specific problems
    void Uniform  (field_array &v, field_array &u);
    void Alfven   (field_array &v, field_array &u);
    void BlastWave(field_array &v, field_array &u);

  private:
    Parameters &config;
    sycl::queue qq;
    TB::Timer stepTime_;
    field tMax_, t_, dt_, cfl_, tOut_;
    unsigned int N_, nxNH_, nyNH_, nzNH_;
    unsigned long BOVRank_; // Necessary as BOV output assumes zyx output order
    unsigned long iOut_, iStep_, nStep_;
    Grid   *grid_;
    Domain *D_;
};

#endif
