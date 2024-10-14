# Copyright(C) 2020 Salvatore Cielo, Leibniz Rechenzentrum
# Copyright(C) 2021 Alexander Pöppl, Intel Corp.
#
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at          http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.22)
include(CheckIncludeFileCXX)

project( DPEcho
  DESCRIPTION "Rewrite of Echo in DPC++"
  LANGUAGES CXX
)

add_executable (dpecho
  echo.cpp Logger.cpp Grid.cpp Domain.cpp Problem.cpp Physics.cpp Metric.cpp Solver.cpp Output.cpp Device.cpp Parameters.cpp
  echo.hpp Logger.hpp Grid.hpp Domain.hpp Problem.hpp Physics.hpp Metric.hpp Solver.hpp Output.hpp Device.hpp Parameters.hpp
)
set_target_properties(dpecho PROPERTIES OUTPUT_NAME "dpecho")
set_property(TARGET dpecho PROPERTY CXX_STANDARD 17)

# Add a folder for CMake modules... (e.g. to find external libraries)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

# -- Build
set(MYBUILDTYPE Release Debug RelWithDebInfo MinSizeRel)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${MYBUILDTYPE})

# TODO
# Missing:
# - support AdaptiveCPP again
# - support for Intel FPGAs

include(CheckLanguage)
check_language(CUDA)
check_language(HIP)

set(TARGET_GPU_AMD "gfx90a" CACHE STRING "Set the type of AMD GPU that will be targeted.")
set_property(CACHE TARGET_GPU_AMD PROPERTY STRINGS gfx900 gfx906 gfx908 gfx90a gfx942)

set(TARGET_GPU_NVIDIA "sm_80" CACHE STRING "Set the type of NVIDIA GPU that will be targeted.")
set_property(CACHE TARGET_GPU_NVIDIA PROPERTY STRINGS sm_60 sm_70 sm_75 sm_80 sm_89 sm_90 sm_95)

set(TARGET_SYCL_BACKENDS "")

if (CMAKE_HIP_COMPILER)
  message(STATUS "Adding AMDGCN backend.")
  list(APPEND TARGET_SYCL_BACKENDS "amdgcn-amd-amdhsa")
  target_compile_options(dpecho PUBLIC -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${TARGET_GPU_AMD})
  target_link_libraries (dpecho PUBLIC -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${TARGET_GPU_AMD})
endif()

if (CMAKE_CUDA_COMPILER)
  message(STATUS "Adding NVPTX64 backend.")
  list(APPEND TARGET_SYCL_BACKENDS "nvptx64-nvidia-cuda")
  target_compile_options(dpecho PUBLIC -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${TARGET_GPU_NVIDIA} -mllvm -nvptx-sched4reg=true)
  target_link_libraries (dpecho PUBLIC -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${TARGET_GPU_NVIDIA} -mllvm -nvptx-sched4reg=true)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  message(STATUS "Adding Intel Level Zero backend.")
  list(APPEND TARGET_SYCL_BACKENDS "spir64")
  target_link_libraries(dpecho PUBLIC
    $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,2024.1>>:-ftarget-register-alloc-mode=pvc:large>)
endif()

string(REPLACE ";" "," TARGET_SYCL_BACKENDS "${TARGET_SYCL_BACKENDS}")

# Support for acpp
if(CMAKE_CXX_COMPILER MATCHES "acpp")
  find_package(AdaptiveCpp CONFIG)
  if(AdaptiveCpp_FOUND)
    message(STATUS "Setting acpp as compiler.")
  else()
    message(ERROR  "acpp compiler requested, but AdaptiveCpp not found. Loaded it?")
  endif(AdaptiveCpp_FOUND)
else()
  target_compile_options(dpecho PUBLIC -fsycl -fsycl-targets=${TARGET_SYCL_BACKENDS})
  target_link_libraries (dpecho PUBLIC -fsycl -fsycl-targets=${TARGET_SYCL_BACKENDS})
endif()

#- VTune support, transparent to the user.
find_package(VTune)
if (VTune_FOUND)
 target_include_directories(dpecho PRIVATE ${VTune_INCLUDE_DIRS})
  target_link_libraries(dpecho PUBLIC ${VTune_LIBRARIES})
  target_compile_definitions(dpecho PUBLIC VTUNE_API_AVAILABLE)
  message(STATUS "Enabling VTune API integration for Echo. More fine-grained profiling available.")
else()
  message(STATUS "VTune Profiler unavailable. Consider making it available at configuration time it if you plan on using VTune.")
endif(VTune_FOUND)

#-- What to do for MPI-Parallel code
set(MPI OFF CACHE STRING "Select the mode of parallel execution. OFF for buildig w/o MPI. For MPI builds, SR_REPLACE is best tested.")
set(MPI_VALS  OFF SR_REPLACE SENDRECV ISEND START)
set_property(CACHE MPI PROPERTY STRINGS ${MPI_VALS})
if (NOT MPI STREQUAL "OFF")
  find_package(MPI)
  if (MPI_FOUND)
    string(TOLOWER ${MPI} MYMPI)
    set(SUFFIX "${SUFFIX}_${MYMPI}")
    target_link_libraries(dpecho PUBLIC ${MPI_CXX_LIBRARIES})
    target_include_directories(dpecho PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_compile_definitions(dpecho PRIVATE MPICODE=${MPI})
    target_compile_options(dpecho PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
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

set(WARNINGS REASONABLE CACHE STRING "Select the warning level.")
set(WARNINGS_VALS MINIMAL REASONABLE PEDANTIC EXCESSIVE)
set_property(CACHE WARNINGS PROPERTY STRINGS ${WARNINGS_VALS})
if (WARNINGS STREQUAL "MINIMAL")
  message(STATUS "Compiler warnings are mostly disabled.")
elseif (WARNINGS STREQUAL "REASONABLE")
  message(STATUS "Reasonable set of compiler warnings enabled.")
  target_compile_options(dpecho PUBLIC -Wuninitialized -Wall -Wextra -Wsign-compare -Wno-unused-parameter -Wno-unused-variable)
elseif (WARNINGS STREQUAL "PEDANTIC")
  message(STATUS "Extensive set of compiler warnings enabled.")
  target_compile_options(dpecho PUBLIC -Wall -Wextra -pedantic)
elseif (WARNINGS STREQUAL "EXCESSIVE")
  message(STATUS "ALL POSSIBLE compiler warnings enabled.")
  target_compile_options(dpecho PUBLIC -Wall -Wextra -pedantic -Weverything)
else()
endif()

# -- DPC++ and Runtime
set(SYCL_DEVICE GPU CACHE STRING "Select the primary SYCL device (default/CPU/GPU/etc.). Overridden by forceDevice in dpecho.par")
set(MYDEVICE DEF CPU GPU ACC FPGA)
set_property(CACHE SYCL_DEVICE PROPERTY STRINGS ${MYDEVICE})
target_compile_definitions(dpecho PRIVATE DEVICE=DEV_${SYCL_DEVICE})
if(NOT SYCL_DEVICE STREQUAL DEF )
  string(TOLOWER ${SYCL_DEVICE} DEVSUFFIX)
  set(SUFFIX "${SUFFIX}_${DEVSUFFIX}")
endif()

# -- Compiling
option(ENABLE_MEMORY_SANITATION "Enable memory sanitizer to find and track invalid memory accesses. Currently disabled." OFF)
if (ENABLE_MEMORY_SANITATION)
  message(STATUS "Memory sanitation enabled. Linking appropriate libraries.")
  target_compile_options(dpecho PUBLIC -fno-omit-frame-pointer -Wuninitialized -g -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope)
  target_link_libraries (dpecho PUBLIC -fno-omit-frame-pointer -g -fsanitize=address -fsanitize=undefined)
endif()

option(SINGLE_PRECISION "Change field from double to float" OFF)
if(SINGLE_PRECISION)
  set(SUFFIX "${SUFFIX}_sp")
  target_compile_definitions(dpecho PUBLIC SINGLE_PRECISION)
  target_compile_options(dpecho PUBLIC -cl-single-precision-constant)
  message(STATUS "Compilation with single precision floating point numbers enabled.")
else()
  message(STATUS "Compilation with double precision floating point numbers enabled.")
endif(SINGLE_PRECISION)


find_package(Boost)
option(ENERGY_METER "Provide ./deltaEnergy.sh with user-defined command in run folder to print used energy." OFF)
if(Boost_FOUND AND ENERGY_METER)
  set(SUFFIX "${SUFFIX}_nrg")
  target_compile_definitions(dpecho PUBLIC TB_ENERGY)
  target_include_directories(dpecho PUBLIC ${Boost_INCLUDE_DIRS})
  target_link_libraries(dpecho PUBLIC Boost::boost -pthread)
  message(STATUS "ENERGY_METER is ON: User *MUST* ensure deltaEnergy.sh script is in working order.")
elseif(ENERGY_METER)
  message(WARNING "ENERGY_METER is requested but relies on Boost Process to be available. Disabling Energy collection.")
endif()

# -- Code
option(UCT "Constrained Transport" OFF)
if(UCT)
  target_compile_definitions(dpecho PRIVATE CT)
endif()

set(FILE_IO_VALUES VISIT_BOV DISABLED)
set(FILE_IO DISABLED CACHE STRING "Type of File IO")
set_property(CACHE FILE_IO PROPERTY STRINGS ${FILE_IO_VALUES})
message(STATUS "File IO is set to: ${FILE_IO}")
target_compile_definitions(dpecho PUBLIC FILE_IO_${FILE_IO})

set(METRIC_VALUES CARTESIAN KERR_SCHILD)
set(METRIC CARTESIAN CACHE STRING "Type of metric used for the simulation.")
set_property(CACHE METRIC PROPERTY STRINGS ${METRIC_VALUES})
if(METRIC STREQUAL KERR_SCHILD)
  message(STATUS "The Kerr-Schild Metric is not tested (may be incorrect!) and has no proper Problem case. It is expected to fail!")
endif()
target_compile_definitions(dpecho PUBLIC METRIC=${METRIC})

set(NRK_VALUES 1 2 3)
set(NRK 3 CACHE STRING "Order of RK scheme")
set_property(CACHE NRK PROPERTY STRINGS ${NRK_VALUES})
target_compile_definitions(dpecho PUBLIC NRK=${NRK})

set(FD_VALUES 2 4 6)
set(FD 6 CACHE STRING "Order of derivation scheme")
set_property(CACHE FD PROPERTY STRINGS ${FD_VALUES})
target_compile_definitions(dpecho PUBLIC FD=${FD})

set(REC_ORDER_VALUES 1 2 5)
set(REC_ORDER 5 CACHE STRING "Order of reconstruction scheme")
set_property(CACHE REC_ORDER PROPERTY STRINGS ${REC_ORDER_VALUES})
target_compile_definitions(dpecho PUBLIC REC_ORDER=${REC_ORDER})

if   (REC_ORDER LESS_EQUAL 2)
  set(RECTYPE NO LINEAR MINMOD MONCEN VANLEER)
  set(RECONSTR VANLEER CACHE STRING "Affects holibRec")
  set_property(CACHE RECONSTR PROPERTY STRINGS ${RECTYPE})
  target_compile_definitions(dpecho PUBLIC RECONSTR=${RECONSTR})
else ()
  unset(RECONSTR CACHE)
endif(REC_ORDER LESS_EQUAL 2)

set(PHYSTYPE MHD GRMHD)
set(PHYSICS GRMHD CACHE STRING "Choose Physics Solver")
set_property(CACHE PHYSICS PROPERTY STRINGS ${PHYSTYPE})
target_compile_definitions(dpecho PUBLIC PHYSICS=${PHYSICS})
if (PHYSICS STREQUAL GRMHD)
  set(SUFFIX ${SUFFIX}_gr)
endif()

##-- Executable name
set_target_properties(dpecho PROPERTIES OUTPUT_NAME "dpecho${SUFFIX}")
