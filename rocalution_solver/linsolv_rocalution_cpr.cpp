#include "linsolv_rocalution_cpr.hpp"

#include "block_csr_matrix.hpp"
#include "linsolv_rocalution.hpp"
#include "scalar_csr_adapter.hpp"
#include "solver_config.hpp"

#include <cmath>
#include <memory>

namespace opendarts
{
  namespace linear_solvers
  {
    struct linsolv_rocalution_cpr::Impl
    {
      block_csr_matrix *A = nullptr;
      std::unique_ptr<scalar_csr_adapter> scalar;
      linsolv_rocalution solver;
      int max_iters = 50;
      double tolerance = 1e-6;
      bool ready = false;

      Impl()
        : solver(0, 0)
      {
      }
    };

    linsolv_rocalution_cpr::linsolv_rocalution_cpr()
      : impl(new Impl)
    {
    }

    linsolv_rocalution_cpr::~linsolv_rocalution_cpr()
    {
      delete impl;
    }

    int linsolv_rocalution_cpr::set_prec(linear_solver *prec_input)
    {
      (void)prec_input;
      return 0;
    }

    int linsolv_rocalution_cpr::reconfigure(const solver_config &config)
    {
      if (config.max_iterations > 0)
        impl->max_iters = config.max_iterations;
      if (config.tolerance > 0.0)
        impl->tolerance = config.tolerance;
      return 0;
    }

    int linsolv_rocalution_cpr::init(csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance)
    {
      auto *block = dynamic_cast<block_csr_matrix *>(A);
      if (block == nullptr || block->block_size() < 2)
        return -1;
      impl->A = block;
      impl->scalar = std::make_unique<scalar_csr_adapter>(*block);
      impl->solver.set_preconditioner(3, block->block_size());
      if (max_iters > 0)
        impl->max_iters = max_iters;
      if (tolerance > 0.0)
        impl->tolerance = tolerance;
      impl->ready = false;
      return 0;
    }

    int linsolv_rocalution_cpr::setup(csr_matrix_base *A)
    {
      if (A != nullptr && A != impl->A)
      {
        const int rc = init(A, impl->max_iters, impl->tolerance);
        if (rc != 0)
          return rc;
      }
      if (impl->A == nullptr || impl->scalar == nullptr)
        return -1;
      impl->scalar->refresh();
      const int rc = impl->solver.init(impl->scalar->n_rows(), impl->scalar->row_ptr(),
          impl->scalar->col_ind(), impl->scalar->values(), impl->max_iters, impl->tolerance);
      if (rc != 0)
        return rc;
      impl->ready = impl->solver.setup() == 0;
      return impl->ready ? 0 : -1;
    }

    int linsolv_rocalution_cpr::solve(opendarts::config::mat_float *B,
        opendarts::config::mat_float *X)
    {
      if (!impl->ready || B == nullptr || X == nullptr)
        return -1;
      const int solved = impl->solver.solve(B, X);
      const int n = impl->scalar->n_rows();
      const auto *row = impl->scalar->row_ptr();
      const auto *col = impl->scalar->col_ind();
      const auto *val = impl->scalar->values();
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
      if (rel <= std::max(impl->tolerance * 50.0, 1.0e-4))
        return 0;
      return solved == 0 ? -1 : solved;
    }

    int linsolv_rocalution_cpr::get_n_iters()
    {
      return impl->solver.get_n_iters();
    }

    opendarts::config::mat_float linsolv_rocalution_cpr::get_residual()
    {
      return impl->solver.get_residual();
    }
  } // namespace linear_solvers
} // namespace opendarts
