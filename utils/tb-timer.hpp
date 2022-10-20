//  Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//  Copyright(C) 2022 Salvatore Cielo, LRZ
//  Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _TOOLBOX_TIMER_H_
#define _TOOLBOX_TIMER_H_

#ifdef MPICODE
#include <mpi.h>
#else
#include <sys/time.h>
#endif

namespace TB {

  class Timer{
    private:
      double t0, sum;
#ifndef MPICODE
      struct timeval TT;
#endif

      inline double get() {
#ifdef MPICODE
	return MPI_Wtime();
#else
	gettimeofday(&TT, (struct timezone *) NULL);
	return (TT.tv_sec)+(TT.tv_usec)*static_cast<double>(0.000001);
#endif
      }

    public:
      Timer() {};
      inline void   init(){ sum= 0.0; on(); }
      inline void   on  (){ t0 = get(); }
      inline double lap (){ double tt=get(), tr=tt-t0; t0=tt; sum+=tr;  return tr; }
      inline double tot (){ return sum; }
  }; //  Timer
} // TB

#endif
