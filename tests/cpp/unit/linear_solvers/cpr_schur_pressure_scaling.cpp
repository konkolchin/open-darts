#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef WITH_GPU
#include <cuda_runtime.h>
#include "csr_matrix.hpp"
#include "linsolv_cpr_gpu.hpp"
#include "linsolv_schur_elim.hpp"

using opendarts::config::index_t;
using opendarts::config::mat_float;
using namespace opendarts::linear_solvers;

namespace
{
  void check_cuda(cudaError_t status)
  {
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }

  class checked_stage : public linear_solver
  {
  public:
    checked_stage(bool pressure, bool device, bool normalized = true)
      : pressure_(pressure), device_(device), normalized_(normalized) {}
    int set_prec(linear_solver *) override { return 0; }
    int init(csr_matrix_base *, index_t, mat_float) override { return 0; }
    int get_n_iters() override { return 0; }
    mat_float get_residual() override { return 0; }

    int setup(csr_matrix_base *matrix) override
    {
      if (!pressure_)
        return 0;
      std::array<mat_float, 7> actual;
      if (device_)
        check_cuda(cudaMemcpy(actual.data(), matrix->get_values_d(), sizeof(actual), cudaMemcpyDeviceToHost));
      else
        std::copy_n(matrix->get_values(), actual.size(), actual.begin());
      if (normalized_)
      {
        // normalize_pressure_rows on: every row divided by its diagonal, so
        // the operator is the SAME whatever per-row scale the Schur
        // condensation gave it.
        const std::array<mat_float, 7> expected = {1, -0.25, -0.25, 1, -0.25, -0.25, 1};
        for (std::size_t entry = 0; entry < actual.size(); ++entry)
          if (std::abs(actual[entry] - expected[entry]) > 1e-12)
          {
            std::cerr << "pressure entry " << entry << ": " << actual[entry]
                      << " expected " << expected[entry] << std::endl;
            return -1;
          }
        return 0;
      }
      // default (sign only): the natural row scale is kept, so only the
      // diagonal's sign and the off-diagonal/diagonal ratio are pinned.
      const std::array<std::size_t, 4> row_begin = {0, 2, 5, 7};
      const std::array<std::size_t, 3> diagonal = {0, 3, 6};
      for (std::size_t row = 0; row < 3; ++row)
      {
        const mat_float d = actual[diagonal[row]];
        if (!(d > 0))
        {
          std::cerr << "pressure diagonal " << row << " not positive: " << d << std::endl;
          return -1;
        }
        for (std::size_t entry = row_begin[row]; entry < row_begin[row + 1]; ++entry)
          if (entry != diagonal[row] && std::abs(actual[entry] / d + 0.25) > 1e-12)
          {
            std::cerr << "pressure entry " << entry << " / diagonal: " << actual[entry] / d
                      << " expected -0.25" << std::endl;
            return -1;
          }
      }
      return 0;
    }

    int solve(mat_float *rhs, mat_float *solution) override
    {
      if (!pressure_)
      {
        check_cuda(cudaMemset(solution, 0, 6 * sizeof(mat_float)));
        return 0;
      }
      const std::array<mat_float, 3> pressure_solution = {1, 2, 3};
      std::array<mat_float, 3> actual_rhs;
      if (device_)
        check_cuda(cudaMemcpy(actual_rhs.data(), rhs, sizeof(actual_rhs), cudaMemcpyDeviceToHost));
      else
        std::copy_n(rhs, actual_rhs.size(), actual_rhs.begin());
      // The restricted residual must carry the SAME multiplier as the matrix
      // rows, whichever scaling mode is in force.
      if (normalized_)
      {
        const std::array<mat_float, 3> expected_rhs = {0.5, 1, 2.5};
        for (std::size_t row = 0; row < actual_rhs.size(); ++row)
          if (std::abs(actual_rhs[row] - expected_rhs[row]) > 1e-12)
            return -1;
      }
      if (device_)
        check_cuda(cudaMemcpy(solution, pressure_solution.data(), sizeof(pressure_solution), cudaMemcpyHostToDevice));
      else
        std::copy(pressure_solution.begin(), pressure_solution.end(), solution);
      return 0;
    }

  private:
    bool pressure_;
    bool device_;
    bool normalized_;
  };

  bool check_schur_cpr(bool pressure_on_device, bool normalize)
  {
    csr_matrix<3> matrix;
    matrix.init(3, 3, 7);
    const std::array<index_t, 4> rows = {0, 2, 5, 7};
    const std::array<index_t, 7> columns = {0, 1, 0, 1, 2, 1, 2};
    const std::array<index_t, 3> diagonals = {0, 3, 6};
    std::copy(rows.begin(), rows.end(), matrix.rows_ptr.begin());
    std::copy(columns.begin(), columns.end(), matrix.cols_ind.begin());
    matrix.diag_ind.resize(3);
    std::copy(diagonals.begin(), diagonals.end(), matrix.diag_ind.begin());
    matrix.init_device(3, 7);
    matrix.copy_struct_to_device();

    timer_node setup_timer, solve_timer;
    auto *cpr = new linsolv_cpr_gpu<2>();
    cpr->p_solver_setup_gpu = pressure_on_device;
    cpr->p_solver_solve_gpu = pressure_on_device;
    cpr->normalize_pressure_rows = normalize ? 1 : 0;
    cpr->set_p_system_prec(new checked_stage(true, pressure_on_device, normalize));
    cpr->set_prec(new checked_stage(false, true));
    linsolv_schur_elim<3, 1> schur(true, {0}, {1});
    schur.set_prec(cpr);
    schur.set_inner_owned(true);
    schur.init_timer_nodes(&setup_timer, &solve_timer);
    if (schur.init(&matrix, 20, 1e-10) != 0)
      return false;

    mat_float *rhs_device = nullptr, *solution_device = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&rhs_device), 9 * sizeof(mat_float)));
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&solution_device), 9 * sizeof(mat_float)));
    bool success = true;
    for (int refresh = 0; refresh < 2; ++refresh)
    {
      // Eight orders of magnitude in both signs. The range is bounded by
      // cpr_weight_max: these blocks give True-IMPES weights of order
      // 1/scale, so a smaller scale than ~1e-6 would (correctly) trip the
      // weight bound and drop the cell to quasi-IMPES, which is a different
      // behaviour from the row scaling under test here.
      const std::array<mat_float, 3> scales = refresh == 0
          ? std::array<mat_float, 3>{1e-4, -3, 1e4}
          : std::array<mat_float, 3>{-1e4, 1e-4, -7};
      const std::array<mat_float, 3> pressure_rhs = {2, 4, 10};
      std::array<mat_float, 9> rhs, solution;
      std::fill(matrix.values.begin(), matrix.values.end(), 0);
      for (index_t row = 0; row < 3; ++row)
      {
        rhs[row * 3] = 3;
        rhs[row * 3 + 1] = scales[row] * (pressure_rhs[row] + 2.5);
        rhs[row * 3 + 2] = 1;
        for (index_t entry = rows[row]; entry < rows[row + 1]; ++entry)
        {
          mat_float *block = matrix.values.data() + entry * 9;
          if (columns[entry] == row)
          {
            block[1] = 1;
            block[3] = 4 * scales[row];
            block[4] = 0.5 * scales[row];
            block[5] = 2 * scales[row];
            block[8] = 2;
          }
          else
            block[3] = -scales[row];
        }
      }
      matrix.copy_values_to_device();
      check_cuda(cudaMemcpy(rhs_device, rhs.data(), sizeof(rhs), cudaMemcpyHostToDevice));
      success = schur.setup(&matrix) == 0 && schur.solve(rhs_device, solution_device) == 0;
      if (!success)
        break;
      check_cuda(cudaMemcpy(solution.data(), solution_device, sizeof(solution), cudaMemcpyDeviceToHost));
      for (index_t row = 0; row < 3; ++row)
        success = success && std::abs(solution[row * 3] - (row + 1)) < 1e-12
            && std::abs(solution[row * 3 + 1] - 3) < 1e-12
            && std::abs(solution[row * 3 + 2]) < 1e-12;
      if (!success)
        break;
    }
    check_cuda(cudaFree(rhs_device));
    check_cuda(cudaFree(solution_device));
    return success;
  }
}
#endif

int main()
{
#ifdef WITH_GPU
  int devices = 0;
  if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0)
  {
    const bool success = check_schur_cpr(false, true) && check_schur_cpr(true, true)
        && check_schur_cpr(false, false) && check_schur_cpr(true, false);
    std::cout << "Schur/CPR pressure and RHS scaling: " << (success ? "PASS" : "FAIL") << std::endl;
    return success ? 0 : 1;
  }
#endif
  std::cout << "Schur/CPR GPU pressure scaling: SKIP (CUDA device unavailable)" << std::endl;
  return 0;
}
