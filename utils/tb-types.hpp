//   Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _TOOLBOX_TYPE_H_
#define _TOOLBOX_TYPE_H_

#include "../echo.hpp"
#include <mpi.h>

#ifdef SINGLE_PRECISION
typedef float real;
#undef MPI_REAL
#define MPI_REAL MPI_FLOAT
constexpr const char* FIELD_FORMAT = "FLOAT";
#else
typedef double real;
#undef MPI_REAL
#define MPI_REAL MPI_DOUBLE
constexpr const char* FIELD_FORMAT = "DOUBLE";
#endif


typedef real coord[3];
typedef unsigned int icoord[3];

using real_array = real *const[FLD_TOT];

#endif
