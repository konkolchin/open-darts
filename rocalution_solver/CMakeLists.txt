# Regression suites for the rocALUTION wrapper.
# Added by the top-level project when WITH_ROCALUTION=ON.
#
# hipcc compiles both programs. They link librocalution plus the
# linsolv_rocalution object stored in the linear_solvers archive.
# open-DARTS compiles the rest of that archive with
# _GLIBCXX_USE_CXX11_ABI=0; rocALUTION uses the default ABI, so these
# targets do not inherit opendarts_compiler_flags.

if(NOT WITH_ROCALUTION)
  return()
endif()

function(od_add_rocalution_test target source)
  add_executable(${target} ${source})
  set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
  # Archive first, then the shared rocALUTION library, so a single-pass
  # link resolves symbols from linsolv_rocalution.o.
  target_link_libraries(${target}
    PRIVATE
      "$<TARGET_LINKER_FILE:linear_solvers>"
      rocalution::rocalution)
  add_dependencies(${target} linear_solvers)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

od_add_rocalution_test(test_linsolv_rocalution test_linsolv_rocalution.cpp)
od_add_rocalution_test(test_linsolv_rocalution_spd test_linsolv_rocalution_spd.cpp)
