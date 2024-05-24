//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _echo_hpp_
#define _echo_hpp_

#define _USE_MATH_DEFINES
#define MAX(x,y) ((x>y)?x:y)

#define NDIM 3


//-- Field numbering
#ifndef FLD_TOT
#define RH 0
#define VX 1
#define VY 2
#define VZ 3
#define PG 4
#define BX 5
#define BY 6
#define BZ 7
#define FLD_VEC 2 // Vector fields
#define FLD_SCA 2 // Scalar fields
#define FLD_TOT 8 // (FLD_SCA + 3*FLD_VEC)  Tot fields
#endif

#ifdef MPICODE
#define SR_REPLACE 1
#define SENDRECV 2
#define ISEND 3
#define START 4
#endif

//-- Holib stuff
#ifdef  RECONSTR
#define NO 0
#define LINEAR 1
#define MINMOD 2
#define MONCEN 3
#define VANLEER 4
#endif

//-- Boundary conditions
#define BCPER 1 // Periodic
#define BCOF0 2 // Outflow w 0th order interpolation
#define BCOF3 3 // Outflow w 3rd order interpolation
#define BCUNK 5 // Unknown

//-- EoS
#define GAMMA 1.3333
#define GAMMA1 4.0 // GAMMA/(GAMMA-1.0)
#define PGFLOOR 0
#define ISOENTROPIC 0

//-- Physics
#define MHD 0
#define GRMHD 1

//-- Metric
#define CARTESIAN 0
#define KERR_SCHILD 1

//-- Numerics
#if REC_ORDER==2
#define REC_LEFT_OFFSET 1
#define REC_RIGHT_OFFSET 2
#define REC_TOTAL_POINTS 4
#elif REC_ORDER==5
#define REC_LEFT_OFFSET 2
#define REC_RIGHT_OFFSET 3
#define REC_TOTAL_POINTS 6
#else
#define REC_LEFT_OFFSET 0
#define REC_RIGHT_OFFSET 0
#define REC_TOTAL_POINTS 1
#endif

#define NGC MAX( (FD/2)+1,(REC_RIGHT_OFFSET)+1 )

#endif
