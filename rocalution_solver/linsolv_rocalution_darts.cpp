#include "linsolv_rocalution_darts.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "block_csr_matrix.hpp"
#include "scalar_csr_adapter.hpp"
#include "solver_config.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    linsolv_rocalution_solver::linsolv_rocalution_solver()
      : solver_(0, 0)
    {
    }

    linsolv_rocalution_solver::~linsolv_rocalution_solver() = default;

    int linsolv_rocalution_solver::set_prec(linear_solver *prec_input)
    {
      (void)prec_input;
      return 0;
    }

    int linsolv_rocalution_solver::reconfigure(const solver_config &config)
    {
      if (config.max_iterations > 0)
        max_iters_ = config.max_iterations;
      if (config.tolerance > 0.0)
        tolerance_ = config.tolerance;
      return 0;
    }

    int linsolv_rocalution_solver::init(csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance)
    {
      auto *block = dynamic_cast<block_csr_matrix *>(A);
      if (block == nullptr || block->block_size() < 1)
        return -1;
      A_ = block;
      scalar_ = std::make_unique<scalar_csr_adapter>(*block);
      if (max_iters > 0)
        max_iters_ = max_iters;
      if (tolerance > 0.0)
        tolerance_ = tolerance;
      ready_ = false;
      return 0;
    }

    int linsolv_rocalution_solver::handoff()
    {
      if (A_ == nullptr || scalar_ == nullptr)
        return -1;
      scalar_->refresh();
      const int n = scalar_->n_rows();
      const auto *row = scalar_->row_ptr();
      const auto *col = scalar_->col_ind();
      const auto *val = scalar_->values();
      const int nnz = scalar_->nnz();
      std::vector<double> scaled(val, val + nnz);
      row_scale_.assign(static_cast<std::size_t>(n), 1.0);
      for (int i = 0; i < n; ++i)
      {
        double max_abs = 0.0;
        for (int k = row[i]; k < row[i + 1]; ++k)
          max_abs = std::max(max_abs, std::abs(scaled[static_cast<std::size_t>(k)]));
        if (max_abs > 0.0)
        {
          const double scale = 1.0 / max_abs;
          row_scale_[static_cast<std::size_t>(i)] = scale;
          for (int k = row[i]; k < row[i + 1]; ++k)
            scaled[static_cast<std::size_t>(k)] *= scale;
        }
      }
      const int rc = solver_.init(n, row, col, scaled.data(), max_iters_, tolerance_);
      if (rc != 0)
        return rc;
      ready_ = solver_.setup() == 0;
      return ready_ ? 0 : -1;
    }

    int linsolv_rocalution_solver::setup(csr_matrix_base *A)
    {
      if (A != nullptr && A != A_)
      {
        const int rc = init(A, max_iters_, tolerance_);
        if (rc != 0)
          return rc;
      }
      return handoff();
    }

    int linsolv_rocalution_solver::solve(opendarts::config::mat_float *B,
        opendarts::config::mat_float *X)
    {
      if (!ready_ || B == nullptr || X == nullptr)
        return -1;
      scaled_b_.resize(row_scale_.size());
      for (std::size_t i = 0; i < row_scale_.size(); ++i)
        scaled_b_[i] = B[i] * row_scale_[i];
      return solver_.solve(scaled_b_.data(), X);
    }

    int linsolv_rocalution_solver::get_n_iters()
    {
      return solver_.get_n_iters();
    }

    opendarts::config::mat_float linsolv_rocalution_solver::get_residual()
    {
      return solver_.get_residual();
    }
  } // namespace linear_solvers
} // namespace opendarts
