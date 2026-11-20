cmake_minimum_required(VERSION 3.18)
project(port_kernel_time LANGUAGES NONE)

# Standalone HIP check. Does not link opendarts_hip_device or the CPU solver.
#   cmake -S rocalution_solver/port_timing -B build-port-time -DOD_PORT_ARCH=gfx1100
# On the RTX 4080 the binary is hipcc with HIP_PLATFORM=nvidia and OD_PORT_ARCH=sm_89.
# hip_compat stays on the include path so hipcc can parse the existing device
# methods in the shared translation units. The timing harness calls the HIP runtime.

if(OD_PORT_LANG AND NOT OD_PORT_LANG STREQUAL "HIP")
  message(FATAL_ERROR "port_timing builds with hipcc only")
endif()
if(NOT OD_PORT_ARCH)
  set(OD_PORT_ARCH "gfx1100")
endif()

set(OD_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(OD_INCLUDES
  -I${OD_ROOT}/rocalution_solver/hip_compat
  -I${CMAKE_CURRENT_SOURCE_DIR}
  -I${OD_ROOT}/linear_solvers/include
  -I${OD_ROOT}/linear_solvers/src
  -I${OD_ROOT}/engines/src
  -I${OD_ROOT}/interpolation/include)

find_program(OD_CC hipcc REQUIRED)
set(OD_ARCH --offload-arch=${OD_PORT_ARCH})
set(OD_XFLAG -x hip)

set(OD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/time_main.cu
  ${OD_ROOT}/engines/src/engine_base_gpu.cpp
  ${OD_ROOT}/linear_solvers/src/linsolv_cpr_gpu.cu
  ${OD_ROOT}/linear_solvers/src/linsolv_schur_elim.cpp
  ${OD_ROOT}/linear_solvers/src/linsolv_block_row_perm.cpp)

set(OD_DEFS -DWITH_GPU -DOD_PORT_BENCH -DOPENDARTS_LINEAR_SOLVERS -O3 -std=c++17)

add_custom_target(test_port_kernel_time ALL
  COMMAND ${OD_CC} ${OD_XFLAG} ${OD_ARCH} ${OD_DEFS} ${OD_INCLUDES} ${OD_SOURCES}
          -o ${CMAKE_CURRENT_BINARY_DIR}/test_port_kernel_time
  DEPENDS ${OD_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/port_bench.h
          ${CMAKE_CURRENT_SOURCE_DIR}/port_bench_time.hpp
  VERBATIM
  COMMENT "test_port_kernel_time (HIP, ${OD_PORT_ARCH})"
)
