#include "linsolv_rocalution_darts.hpp"

#include <cmath>

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
      const int rc = solver_.init(n, scalar_->row_ptr(), scalar_->col_ind(), scalar_->values(),
          max_iters_, tolerance_);
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
      const int solved = solver_.solve(B, X);
      const int n = scalar_->n_rows();
      const auto *row = scalar_->row_ptr();
      const auto *col = scalar_->col_ind();
      const auto *val = scalar_->values();
      double b2 = 0.0;
      double r2 = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double ax = 0.0;
        for (int k = row[i]; k < row[i + 1]; ++k)
          ax += val[k] * X[col[k]];
        const double r = B[i] - ax;
        r2 += r * r;
        b2 += B[i] * B[i];
      }
      if (b2 == 0.0)
        return 0;
      const double rel = std::sqrt(r2 / b2);
      // FGMRES stops on this residual. Still reject a reported success that
      // did not actually reduce it, which previously froze the well residual.
      if (rel <= std::max(tolerance_ * 50.0, 1.0e-4))
        return 0;
      return solved == 0 ? -1 : solved;
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
