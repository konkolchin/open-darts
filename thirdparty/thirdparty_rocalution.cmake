# rocALUTION --------------------------------------------------------------------
# Imports the prebuilt ROCm rocALUTION library (librocalution).
#
# Unlike AMGX, rocALUTION is not a submodule and is not compiled by this
# tree. The ROCm package supplies the headers and the shared library.
# Resolution: ROCM_PATH (cache or environment), then /opt/rocm.
# Defines the imported target rocalution::rocalution.
# ------------------------------------------------------------------------------

message(CHECK_START "   Importing rocALUTION")

if(NOT ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()

find_path(ROCALUTION_INCLUDE_DIR
  NAMES rocalution/rocalution.hpp
  HINTS "${ROCM_PATH}"
  PATH_SUFFIXES include)
find_library(ROCALUTION_LIBRARY
  NAMES rocalution
  HINTS "${ROCM_PATH}"
  PATH_SUFFIXES lib lib64)

if(NOT ROCALUTION_INCLUDE_DIR OR NOT ROCALUTION_LIBRARY)
  message(FATAL_ERROR
    "rocALUTION was not found under ROCM_PATH='${ROCM_PATH}'. "
    "Install the ROCm rocALUTION package or pass -D ROCM_PATH=/path/to/rocm.")
endif()

get_filename_component(ROCALUTION_LIBRARY_DIR "${ROCALUTION_LIBRARY}" DIRECTORY)

if(NOT TARGET rocalution::rocalution)
  add_library(rocalution::rocalution SHARED IMPORTED)
  set_target_properties(rocalution::rocalution PROPERTIES
    IMPORTED_LOCATION "${ROCALUTION_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ROCALUTION_INCLUDE_DIR}"
    INTERFACE_LINK_DIRECTORIES "${ROCALUTION_LIBRARY_DIR}"
    INTERFACE_LINK_OPTIONS "-Wl,-rpath,${ROCALUTION_LIBRARY_DIR}")
endif()

message(STATUS "      Include directory: ${ROCALUTION_INCLUDE_DIR}")
message(STATUS "      Library path     : ${ROCALUTION_LIBRARY}")
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------
