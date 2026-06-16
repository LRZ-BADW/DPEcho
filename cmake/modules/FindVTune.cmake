# Defines:
# VTune_FOUND
# VTune_INCLUDE_DIRS
# VTune_LIBRARIES

set(dirs
  "$ENV{VTUNE_PROFILER_DIR}/"
)

find_path(VTune_INCLUDE_DIRS ittnotify.h
  PATHS ${dirs}
  PATH_SUFFIXES include)

if (CMAKE_SIZEOF_VOID_P MATCHES "8")
  set(vtune_lib_dir lib64)
else ()
  set(vtune_lib_dir lib32)
endif ()

find_library(VTune_LIBRARIES
  NAMES libittnotify.a ittnotify libittnotify
  HINTS "${VTune_INCLUDE_DIRS}/.."
  PATHS ${dirs}
  PATH_SUFFIXES ${vtune_lib_dir}
)


mark_as_advanced(VTune_LIBRARIES)
mark_as_advanced(VTune_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VTune DEFAULT_MSG VTune_LIBRARIES VTune_INCLUDE_DIRS)
