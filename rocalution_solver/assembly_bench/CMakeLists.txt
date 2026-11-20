cmake_minimum_required(VERSION 3.18)
project(assembly_bench LANGUAGES NONE)

# Standalone HIP harness for Hyperloom. Does not link rocALUTION or run a
# timestep. hipcc only:
#   cmake -S rocalution_solver/assembly_bench -B build-assembly-bench -DOD_PORT_ARCH=gfx1100
# The MI300 build is the same sources with -DOD_PORT_ARCH=gfx942.
# hip_compat stays on the include path so hipcc can parse the shared kernel
# sources. The harness calls the HIP runtime.

if(OD_PORT_LANG AND NOT OD_PORT_LANG STREQUAL "HIP")
  message(FATAL_ERROR "assembly_bench builds with hipcc only")
endif()
if(NOT OD_PORT_ARCH)
  set(OD_PORT_ARCH "gfx1100")
endif()

set(OD_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(OD_INCLUDES
  -I${OD_ROOT}/rocalution_solver/hip_compat
  -I${OD_ROOT}/rocalution_solver/port_timing
  -I${OD_ROOT}/linear_solvers/include
  -I${OD_ROOT}/linear_solvers/src
  -I${OD_ROOT}/engines/src
  -I${OD_ROOT}/engines/src/mech
  -I${OD_ROOT}/discretizer/src
  -I${OD_ROOT}/discretizer/src/mesh
  -I${OD_ROOT}/thirdparty/install/include
  -I${OD_ROOT}/interpolation/include
  -I${OD_ROOT}/thirdparty/pybind11/include)

find_program(OD_CC hipcc REQUIRED)
find_program(OD_PY NAMES python3.11 python3 REQUIRED)
execute_process(
  COMMAND ${OD_PY} -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE OD_PY_INC
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND OD_INCLUDES -I${OD_PY_INC})

set(OD_ARCH --offload-arch=${OD_PORT_ARCH})
set(OD_XFLAG -x hip)

set(OD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
  ${OD_ROOT}/linear_solvers/src/linsolv_cpr_gpu.cu)

set(OD_DEFS
  -DWITH_GPU -DOD_PORT_BENCH -DOD_HIP_GPU -DOPENDARTS_LINEAR_SOLVERS
  -DMAX_NC=2 -O3 -std=c++17)

add_custom_target(assembly_bench ALL
  COMMAND ${OD_CC} ${OD_XFLAG} ${OD_ARCH} ${OD_DEFS} ${OD_INCLUDES} ${OD_SOURCES}
          -o ${CMAKE_CURRENT_BINARY_DIR}/assembly_bench
  DEPENDS ${OD_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/../port_timing/port_bench_time.hpp
  VERBATIM
  COMMENT "assembly_bench (HIP, ${OD_PORT_ARCH})"
)
