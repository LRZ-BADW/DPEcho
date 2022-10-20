# Copyright(C) 2020 Salvatore Cielo, Leibniz Rechenzentrum
# Copyright(C) 2021 Alexander Pöppl, Intel Corp.
#
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at          http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.13)
include(CheckIncludeFileCXX)

project( DPEcho
  DESCRIPTION "Rewrite of Echo in DPC++"
  LANGUAGES CXX
)

add_executable (echo
  echo.cpp Logger.cpp Grid.cpp Domain.cpp Problem.cpp Physics.cpp Metric.cpp Solver.cpp Output.cpp DeviceConfig.cpp
  echo.hpp Logger.hpp Grid.hpp Domain.hpp Problem.hpp Physics.hpp Metric.hpp Solver.hpp Output.hpp DeviceConfig.hpp echoSycl.hpp
)
set_target_properties(echo PROPERTIES OUTPUT_NAME "echo")

# Add a folder for CMake modules... (e.g. to find external libraries)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

# -- Build
set(MYBUILDTYPE Release Debug RelWithDebInfo MinSizeRel)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${MYBUILDTYPE})

# Unfortunately, there is no proper support of SyCL in CMake as of now.
# Compilation without passing any environment variables was given as a
# requirement, hence, we set the compiler here directly (CMake discourages this).
set(SYCL oneAPI CACHE STRING "Select the SYCL target architecture")
set(MYSYCL oneAPI oneAPIold LLVM hipSYCL)
set_property(CACHE SYCL PROPERTY STRINGS ${MYSYCL})
target_compile_definitions(echo PRIVATE SYCL=${SYCL})
# TODO: So far the only target for LLVM is NVIDIA, and for hipSYCL omp/cpu. Add the others!
if (SYCL STREQUAL LLVM)
  set(CMAKE_CXX_COMPILER "clang++")
  message(STATUS "Configuring Echo with LLVM (NVIDIA) Toolchain")
  target_compile_options(echo PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
  target_link_options   (echo PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
  message(STATUS "Set device target to NVIDIA PTX64")
elseif (SYCL STREQUAL hipSYCL )
  set(CMAKE_CXX_COMPILER "syclcc-clang")
  message(STATUS "Configuring Echo with hipSycl Toolchain")
else()
  set(CMAKE_CXX_COMPILER "dpcpp")
  message(STATUS "Configuring Echo with LLVM (Intel) Toolchain")
  target_compile_options(echo PUBLIC -fsycl -fsycl-unnamed-lambda -xHost)
  target_link_options   (echo PUBLIC -fsycl)
  message(STATUS "Set device target to SPIR-V Just-in-Time Compilation")
  #-- Stuff specific for oneAPI (for the moment)
  option(ENABLE_MPI "Compile with MPI besides DPC++" OFF)
  #- IMPI
  find_package(VTune)
  #- VTune support, transparent to the user.
  if (VTune_FOUND)
    target_include_directories(echo PRIVATE ${VTune_INCLUDE_DIRS})
    target_link_libraries(echo PUBLIC ${VTune_LIBRARIES})
    target_compile_definitions(echo PUBLIC VTUNE_API_AVAILABLE)
    message(STATUS "Enabling VTune API integration for Echo. More fine-grained profiling available.")
  else()
    message(STATUS "VTune Profiler unavailable. Consider making it available at configuration time it if you plan on using VTune.")
  endif(VTune_FOUND)
endif()

#-- What to do for MPI-Parallel code
if (ENABLE_MPI)
  find_package(MPI)
  if (MPI_FOUND)
    set(SUFFIX "${SUFFIX}_mpi")
    target_link_libraries(echo PUBLIC ${MPI_CXX_LIBRARIES})
    target_include_directories(echo PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_compile_definitions(echo PRIVATE MPICODE)
    target_compile_options(echo PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
    message(STATUS "Configuring Echo with MPI support")
  else()
    message(FATAL_ERROR "MPI enabled but not found")
  endif()
else()
  message(STATUS  "Configuring Echo without MPI")
  if (VTune_FOUND)
    message(STATUS "Trying VTune profiling w/o MPI: calls to MPI_PControl will be disabled!")
  endif()
endif()

option(ENABLE_ALL_WARNINGS "Enable a large set of compiler warnings" OFF)
if (ENABLE_ALL_WARNINGS)
  message(STATUS "Extensive set of compiler warnings enabled.")
  target_compile_options(echo PUBLIC -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -pedantic)
else()
  message(STATUS "Compiler warnings are mostly disabled.")
endif()

# -- DPC++ and Runtime
option(GPU "Uses device selector rather than host" OFF)
if(GPU)
  set(SUFFIX "${SUFFIX}_gpu")
  target_compile_definitions(echo PRIVATE GPU)
endif(GPU)

try_compile(COMPILE_SUCCEEDED
            ${CMAKE_BINARY_DIR}/check
            SOURCES ${CMAKE_SOURCE_DIR}/cmake/checks/fscheck.cpp
            CXX_STANDARD 17
)
if(COMPILE_SUCCEEDED)
  message(STATUS "C++ filesystem API available. Using that.")
else()
  message(STATUS "C++ filesystem API UNavailable. Using internal workaround.")
  target_compile_definitions(echo PRIVATE USE_FILE_SYSTEM_API_WORKAROUND)
endif()
CHECK_INCLUDE_FILE_CXX("filesystem.h" FSH_FOUND)
CHECK_INCLUDE_FILE_CXX(filesystem FS_FOUND)


option(ENABLE_MEMORY_SANITATION "Enable memory sanitizer to find and track invalid memory accesses. Currently disabled." OFF)
#if (ENABLE_MEMORY_SANITATION AND NOT GPU)
#  message(STATUS "Memory sanitation enabled. Linking appropriate libraries.")
#  target_compile_options(echo PUBLIC  -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope -Wuninitialized -g)
#  target_link_libraries(echo PUBLIC  -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -g)
#elseif(ENABLE_MEMORY_SANITATION)
#  message(WARNING "Memory sanitation is only available with disabled GPU for now.")
#endif()

option(SINGLE_PRECISION "Change field from double to float" OFF)
if(SINGLE_PRECISION)
  set(SUFFIX "${SUFFIX}_sp")
  target_compile_definitions(echo PUBLIC SINGLE_PRECISION)
  target_compile_options(echo PUBLIC -cl-single-precision-constant)
  message(STATUS "Compilation with single precision floating point numbers enabled.")
else()
  message(STATUS "Compilation with double precision floating point numbers enabled.")
endif(SINGLE_PRECISION)

# -- Code
option(UCT "Constrained Transport" OFF)
if(UCT)
  target_compile_definitions(echo PRIVATE CT)
endif()

set(FILE_IO_VALUES VISIT_BOV DISABLED)
set(FILE_IO VISIT_BOV CACHE STRING "Type of File IO")
set_property(CACHE FILE_IO PROPERTY STRINGS ${FILE_IO_VALUES})
message(STATUS "File IO is set to: ${FILE_IO}")
target_compile_definitions(echo PUBLIC FILE_IO_${FILE_IO})

set(METRIC_VALUES CARTESIAN KERR_SCHILD)
set(METRIC CARTESIAN CACHE STRING "Type of metric used for the simulation.")
set_property(CACHE METRIC PROPERTY STRINGS ${METRIC_VALUES})
target_compile_definitions(echo PUBLIC METRIC=${METRIC})

set(NRK_VALUES 1 2 3)
set(NRK 2 CACHE STRING "Order of RK scheme")
set_property(CACHE NRK PROPERTY STRINGS ${NRK_VALUES})
target_compile_definitions(echo PUBLIC NRK=${NRK})

set(FD_VALUES 2 4 6)
set(FD 4 CACHE STRING "Order of derivation scheme")
set_property(CACHE FD PROPERTY STRINGS ${FD_VALUES})
target_compile_definitions(echo PUBLIC FD=${FD})

set(REC_ORDER_VALUES 1 2 5)
set(REC_ORDER 5 CACHE STRING "Order of reconstruction scheme")
set_property(CACHE REC_ORDER PROPERTY STRINGS ${REC_ORDER_VALUES})
target_compile_definitions(echo PUBLIC REC_ORDER=${REC_ORDER})

if   (REC_ORDER LESS_EQUAL 2)
  set(RECTYPE NO LINEAR MINMOD MONCEN VANLEER)
  set(RECONSTR VANLEER CACHE STRING "Affects holibRec")
  set_property(CACHE RECONSTR PROPERTY STRINGS ${RECTYPE})
  target_compile_definitions(echo PUBLIC RECONSTR=${RECONSTR})
else ()
  unset(RECONSTR CACHE)
endif(REC_ORDER LESS_EQUAL 2)

set(PHYSTYPE MHD GRMHD)
set(PHYSICS MHD CACHE STRING "Choose Physics Solver")
set_property(CACHE PHYSICS PROPERTY STRINGS ${PHYSTYPE})
target_compile_definitions(echo PUBLIC PHYSICS=${PHYSICS})
if (PHYSICS STREQUAL GRMHD)
  set(SUFFIX ${SUFFIX}_gr)
endif()

##-- Executable name
set_target_properties(echo PROPERTIES OUTPUT_NAME "echo${SUFFIX}")
