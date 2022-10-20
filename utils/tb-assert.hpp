/*
 Copyright(C) 2020 Fabio Baruffa, Intel Corp.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef _TOOLBOX_ASSERT_H_
#define _TOOLBOX_ASSERT_H_

#include <iostream>

#ifdef MPICODE
#include <mpi.h>
#endif

using namespace std;

#ifndef DEBUG
        #define ASSERT(x)
#else
        #define ASSERT(x) \
        TB::logAssert(x, #x, __FILE__, __LINE__)
#endif

namespace TB {

  inline bool logAssert(bool x, const string msg, const string file, unsigned int line)
  {
    if( false == x)
    {
      cout<<" On line " <<line<<":";
      cout<<" in file " <<file <<":";
      cout<<" Error !! Assert "<<msg << " failed\n";
#ifdef MPICODE
      int rv = -1;
      MPI_Abort(MPI_COMM_WORLD,rv);
#else
      abort();
#endif
      return(true);
    }
    else
    {
      return(true);
    }
  }
} 

#endif
