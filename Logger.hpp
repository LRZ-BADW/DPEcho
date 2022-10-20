//   Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//   Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _Logger_hpp_
#define _Logger_hpp_

#define TAG mytag(__PRETTY_FUNCTION__).c_str()
#define KTAG __FILE__ // Usable inside DPC++ kernels. For lack of a better one

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "utils/tb-timer.hpp"
#include "utils/tb-assert.hpp"

#ifdef MPICODE
#include <mpi.h>
#endif

#ifdef VTUNE_API_AVAILABLE
#include <ittnotify.h>
#endif

const string mytag(const string); // Prototype for easy logging

// This class includes the information for the entire application.
// This has been implemented as singleton pattern
class Logger {
  public:
    static Logger* getInstance(int *argc, char*** argv);
    static Logger* getInstance();
    inline void setInfoVerbosity(int v){verb_ = v;}

    void Logo ();
    void Info (int level, const char *msg, ... );
    void Write(const char *msg, ... );
    void Debug(const char *msg, ... );
    void Error(const char *msg, ... );
    void togglePcontrol(int onOff){
#ifndef NDEBUG
#ifdef VTUNE_API_AVAILABLE
      if(!onOff){ __itt_pause (); }
      else      { __itt_resume(); }
#endif
#ifdef MPICODE
      MPI_Pcontrol(onOff);
#endif
#endif
    }
    // Operators & very closedly related functions are defined here

    // C++ style output. Switch from parallel output to stdout to serial into log file
    void setPar(const bool parIO){
      if ( pario_ !=  parIO){ pario_ = parIO; barrier(); }
    }

    template <typename T> Logger& operator<< (const T &val){
      if(cverb_ > verb_) { return *this;}
      if(pario_        ) { barrier(); ofs_ << val; } // Parallel, on stdio
      else if (master()) { ffs_ << val; }            // Master-only, on file
      return *this;
    }

    inline void fl(){   // Use it to flush all outputs!
      if( cverb_ > verb_) { return ;}
      if(pario_) {
        barrier();
        for (unsigned int i = 0; i < nprocs_ ; i++){
          if( i == myrank_ ){ cout << ofs_.str()<<std::endl; }
          barrier();
        }
        ofs_.str(""); ofs_.clear();
      } else if ( master() ){ ffs_ << std::endl<<std::flush; }
    }

    Logger& operator+ (const unsigned int val){ // Sets verbosity of next messages
      unsigned short int j;
      cverb_ = val; barrier();
      if (val > verb_){ return *this; }
      if (pario_     ){ ofs_ <<"["<< getMyRank() <<"]"; }
      return *this;
    }

    inline int  getNumProcs() { return nprocs_; };
    inline int  getMyRank  () { return myrank_; };
    ofstream&   ffs        () { return ffs_   ; };

    bool master();
    void barrier();
    void cleanup();

  protected:
    static bool instanceFlag;
    static Logger *single;
    Logger(int *argc, char*** argv);
    Logger(const Logger&);
    Logger& operator=(const Logger&);
    ~Logger();

  private:
    bool pario_;
    ofstream ffs_;       // File stream for proper log
    ostringstream ofs_;  // "buffer" stream for parallel I/O
    static Logger* instance;
    TB::Timer time;
    int verb_, cverb_;
    int nprocs_;
    int myrank_;
};

#endif
