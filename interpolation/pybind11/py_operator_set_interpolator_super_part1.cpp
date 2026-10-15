#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 1: Thermal single-phase / two-phase, plus the history-augmented specializations
// N_OPS = (2 * NP + 2) * NC + 7 * NP + 3
// OBL history axes (HistoryField, e.g. sg_max) add interpolator dimensions but no operators, so a physics with
// n_history axes needs N_DIMS = N_VARS + n_history at the N_OPS of N_VARS. The super engines read operators with
// their own N_OPS stride, so the higher-N_OPS fallback of PhysicsBase.create_interpolator cannot replace a missing
// family here; every (NP, n_history) combination in use needs its exact (A, B).

void pybind_operator_set_interpolator_super_part1(py::module &m)
{
  const int N_DIMS_MAX = MAX_DIMS;

  // NP = 1: A = 4, B = 10 (thermal single-phase)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 4, 10> e1;

  // NP = 2: A = 6, B = 17 (thermal two-phase)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 6, 17> e2;

  // NP = 2: A = 6, B = 11 (two-phase hysteresis / history-augmented compositional case;
  // N_OPS = 6*(NC-1) + 7*NP + 3)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 6, 11> e3;

  // NP = 3: A = 8, B = 16 (three-phase history-augmented case: one OBL history axis on top of
  // the N_VARS axes, N_OPS = 8*(N_DIMS-1) + 7*NP + 3), e.g. 4_48 (3 components) and 5_56 (4 components)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 8, 16> e4;

  // NP = 3: A = 8, B = 8 (three-phase with two OBL history axes, N_OPS = 8*(N_DIMS-2) + 7*NP + 3)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 8, 8> e5;

  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
  e4.expose(m);
  e5.expose(m);
}

#endif //PYBIND11_ENABLED
