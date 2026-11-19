#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "data_types.hpp"
#include "test_common.hpp"

#if defined(WITH_GPU) && defined(WITH_AMGX)

#include "linsolv_amgx.hpp"
#include "timer_node.h"

// Same 12-unknown tridiagonal system as linsolv_superlu (block size 1, rhs = 1).
// The reference is that test's SuperLU solution. This file calls only linsolv_amgx.

static const char *k_config_name = "amgx_config_bs1.json";

static int write_solve_config()
{
  // 12 rows is below min_coarse_rows, so AMGX hands the system to its dense
  // coarse solver. max_iters stays above 1 so a larger matrix still iterates.
  std::ofstream out(k_config_name);
  if (!out)
    return 1;
  out <<
    "{\n"
    "  \"config_version\": 2,\n"
    "  \"solver\": {\n"
    "    \"solver\": \"AMG\",\n"
    "    \"algorithm\": \"CLASSICAL\",\n"
    "    \"smoother\": \"BLOCK_JACOBI\",\n"
    "    \"coarse_solver\": \"DENSE_LU_SOLVER\",\n"
    "    \"min_coarse_rows\": 256,\n"
    "    \"max_iters\": 20,\n"
    "    \"tolerance\": 1e-10,\n"
    "    \"convergence\": \"ABSOLUTE\",\n"
    "    \"norm\": \"L2\",\n"
    "    \"presweeps\": 1,\n"
    "    \"postsweeps\": 1,\n"
    "    \"print_solve_stats\": 1,\n"
    "    \"monitor_residual\": 1,\n"
    "    \"obtain_timings\": 0\n"
    "  }\n"
    "}\n";
  return out.good() ? 0 : 1;
}

static int run()
{
  const opendarts::config::index_t n_rows = 12;
  const opendarts::config::mat_float error_tol = 1e-8;
  const std::vector<opendarts::config::mat_float> reference{
      -0.071823204419889, -0.071823204419889, 0.535911602209945, 0.535911602209945,
      0.160220994475138, 0.160220994475138, 0.955801104972376, 0.955801104972376,
      0.182320441988950, 0.182320441988950, 1.364640883977901, 1.364640883977901};

  if (write_solve_config() != 0)
  {
    std::cout << "FAIL could not write " << k_config_name << std::endl;
    return 1;
  }

  opendarts::linear_solvers::csr_matrix<1> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
      A, n_rows, opendarts::linear_solvers::testing::block_fill_option::diagonal_block);

  const int nnz = static_cast<int>(A.get_rows_ptr()[A.n_rows]);
  if (A.init_device(static_cast<int>(A.n_rows), nnz) != 0)
    return 1;
  if (A.copy_struct_to_device() != 0 || A.copy_values_to_device() != 0)
    return 1;

  // Constructed after the json exists: the constructor reads amgx_config_bs1.json.
  opendarts::linear_solvers::linsolv_amgx<1> solver(/*device_num=*/0, /*convert_to_bs1=*/1, /*reuse_max_override=*/0);

  ::timer_node timer_setup, timer_setup_amgx;
  ::timer_node timer_solve, timer_solve_amgx;
  timer_setup.node.emplace("AMGX<1>", timer_setup_amgx);
  timer_solve.node.emplace("AMGX<1>", timer_solve_amgx);
  solver.init_timer_nodes(&timer_setup, &timer_solve);

  if (solver.init(&A, 0, 0.0) != 0)
  {
    std::cout << "FAIL init" << std::endl;
    return 1;
  }
  if (solver.setup(&A) != 0)
  {
    std::cout << "FAIL setup" << std::endl;
    return 1;
  }

  std::vector<opendarts::config::mat_float> b(n_rows, 1.0);
  std::vector<opendarts::config::mat_float> x(n_rows, 0.0);
  if (solver.solve(b.data(), x.data()) != 0)
  {
    std::cout << "FAIL solve" << std::endl;
    return 1;
  }

  opendarts::config::mat_float error_max = 0.0;
  for (opendarts::config::index_t i = 0; i < n_rows; ++i)
  {
    const opendarts::config::mat_float d = std::abs(x[i] - reference[i]);
    if (d > error_max)
      error_max = d;
    std::cout << x[i] << std::endl;
  }
  std::cout << "max abs diff " << error_max << std::endl;
  if (error_max > error_tol)
  {
    std::cout << "FAIL" << std::endl;
    return 1;
  }
  std::cout << "OK" << std::endl;
  return 0;
}

int main()
{
  return run();
}

#else

int main()
{
  std::cout << "SKIP linsolv_amgx: this build has no AMGX" << std::endl;
  return 0;
}

#endif
