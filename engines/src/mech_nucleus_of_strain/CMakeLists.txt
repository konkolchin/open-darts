# Build the nucleus-of-strain geomechanics proxy python module (_proxygeomech) ---
# A standalone module: it links nothing from the engines and is installed into
# the darts.physics.geomechanics.nucleus_of_strain package.

set(NUCLEUS_OF_STRAIN_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/nucleus_of_strain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_nucleus_of_strain.cpp
)

set(NUCLEUS_OF_STRAIN_HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/nucleus_of_strain.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nucleus_of_strain_cuda.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/py_global.h
)

pybind11_add_module(_proxygeomech
  ${NUCLEUS_OF_STRAIN_SRC_FILES}
  ${NUCLEUS_OF_STRAIN_HEADER_FILES}
)

target_compile_definitions(_proxygeomech PRIVATE PYBIND11_ENABLED)

target_include_directories(_proxygeomech PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The kernel calls omp_set_num_threads and parallelises over evaluation points,
# so OpenMP is linked in every configuration, including the single-threaded one.
find_package(OpenMP REQUIRED COMPONENTS CXX)

target_link_libraries(_proxygeomech
  PRIVATE
    opendarts_compiler_flags
    OpenMP::OpenMP_CXX
)

if(CUDA)
  target_sources(_proxygeomech PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nucleus_of_strain_cuda.cu)
  target_compile_definitions(_proxygeomech PRIVATE USE_CUDA)
  target_link_libraries(_proxygeomech PRIVATE CUDA::cudart)
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NUCLEUS_OF_STRAIN_SRC_FILES} ${NUCLEUS_OF_STRAIN_HEADER_FILES})

install(TARGETS _proxygeomech
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  LIBRARY DESTINATION ./physics/geomechanics/nucleus_of_strain
  RUNTIME DESTINATION ./physics/geomechanics/nucleus_of_strain
)
# ------------------------------------------------------------------------------
