#include "linsolv_rocalution_cpr.hpp"

#include "block_csr_matrix.hpp"
#include "linsolv_rocalution_backend.hpp"
#include "scalar_csr_adapter.hpp"
#include "solver_config.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace opendarts
{
  namespace linear_solvers
  {
    struct linsolv_rocalution_cpr::Impl
    {
      block_csr_matrix *A = nullptr;
      std::unique_ptr<scalar_csr_adapter> scalar;
      rocalution_pair *pressure = nullptr;
      rocalution_pair *full = nullptr;
      std::vector<int> p_row;
      std::vector<int> p_col;
      std::vector<double> p_val;
      std::vector<double> p_scale;
      std::vector<double> full_scale;
      std::vector<double> scaled_full;
      std::vector<double> p_rhs;
      std::vector<double> p_x;
      std::vector<double> full_x;
      std::vector<double> ax;
      int block_size = 0;
      int pressure_var = 0;
      int max_iters = 50;
      double tolerance = 1e-6;
      int iters = 0;
      double residual = 0.0;
      bool ready = false;

      ~Impl()
      {
        if (pressure != nullptr)
          rocalution_pair_destroy(pressure);
        if (full != nullptr)
          rocalution_pair_destroy(full);
      }
    };

    namespace
    {
      void gather_pressure(const block_csr_matrix &A, int pressure_var,
          std::vector<int> &row, std::vector<int> &col, std::vector<double> &val)
      {
        const int nb = A.block_size();
        const int n = static_cast<int>(A.n_block_rows());
        const auto *rows = A.row_ptr();
        const auto *cols = A.col_ind();
        const auto *values = A.values();
        const int nnzb = rows[n];
        const int slot = pressure_var * nb + pressure_var;
        row.resize(static_cast<std::size_t>(n) + 1);
        col.resize(static_cast<std::size_t>(nnzb));
        val.resize(static_cast<std::size_t>(nnzb));
        for (int i = 0; i <= n; ++i)
          row[static_cast<std::size_t>(i)] = rows[i];
        for (int k = 0; k < nnzb; ++k)
        {
          col[static_cast<std::size_t>(k)] = cols[k];
          val[static_cast<std::size_t>(k)] =
              values[static_cast<std::size_t>(k) * nb * nb + slot];
        }
      }

      void scale_rows(int n, const int *row, std::vector<double> &val, std::vector<double> &scale)
      {
        scale.assign(static_cast<std::size_t>(n), 1.0);
        for (int i = 0; i < n; ++i)
        {
          double max_abs = 0.0;
          for (int k = row[i]; k < row[i + 1]; ++k)
            max_abs = std::max(max_abs, std::abs(val[static_cast<std::size_t>(k)]));
          if (max_abs > 0.0)
          {
            const double s = 1.0 / max_abs;
            scale[static_cast<std::size_t>(i)] = s;
            for (int k = row[i]; k < row[i + 1]; ++k)
              val[static_cast<std::size_t>(k)] *= s;
          }
        }
      }

      void matvec(int n, const int *row, const int *col, const double *val,
          const double *x, double *ax)
      {
        for (int i = 0; i < n; ++i)
        {
          double sum = 0.0;
          for (int k = row[i]; k < row[i + 1]; ++k)
            sum += val[k] * x[col[k]];
          ax[i] = sum;
        }
      }
    } // namespace

    linsolv_rocalution_cpr::linsolv_rocalution_cpr()
      : impl(new Impl)
    {
      impl->pressure = rocalution_pair_create(0, 0);
      impl->full = rocalution_pair_create(1, 0);
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
      impl->block_size = block->block_size();
      impl->scalar = std::make_unique<scalar_csr_adapter>(*block);
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
      gather_pressure(*impl->A, impl->pressure_var, impl->p_row, impl->p_col, impl->p_val);
      const int n_blocks = static_cast<int>(impl->A->n_block_rows());
      const int n_scalar = impl->scalar->n_rows();
      scale_rows(n_blocks, impl->p_row.data(), impl->p_val, impl->p_scale);
      impl->scaled_full.assign(impl->scalar->values(), impl->scalar->values() + impl->scalar->nnz());
      scale_rows(n_scalar, impl->scalar->row_ptr(), impl->scaled_full, impl->full_scale);
      if (rocalution_pair_setup(impl->pressure, n_blocks, impl->p_row.data(),
              impl->p_col.data(), impl->p_val.data(), impl->max_iters, impl->tolerance) != 0)
        return -1;
      if (rocalution_pair_setup(impl->full, n_scalar, impl->scalar->row_ptr(),
              impl->scalar->col_ind(), impl->scaled_full.data(), impl->max_iters,
              impl->tolerance) != 0)
        return -1;
      impl->ready = true;
      return 0;
    }

    int linsolv_rocalution_cpr::solve(opendarts::config::mat_float *B,
        opendarts::config::mat_float *X)
    {
      if (!impl->ready || B == nullptr || X == nullptr)
        return -1;
      const int nb = impl->block_size;
      const int n_blocks = static_cast<int>(impl->A->n_block_rows());
      const int n = impl->scalar->n_rows();
      const int p = impl->pressure_var;
      impl->p_rhs.resize(static_cast<std::size_t>(n_blocks));
      impl->p_x.resize(static_cast<std::size_t>(n_blocks));
      impl->full_x.assign(static_cast<std::size_t>(n), 0.0);
      impl->ax.resize(static_cast<std::size_t>(n));
      for (int i = 0; i < n_blocks; ++i)
        impl->p_rhs[static_cast<std::size_t>(i)] =
            B[i * nb + p] * impl->p_scale[static_cast<std::size_t>(i)];
      if (rocalution_pair_solve(impl->pressure, impl->p_rhs.data(), impl->p_x.data()) != 0)
        return -1;
      for (int i = 0; i < n_blocks; ++i)
        impl->full_x[static_cast<std::size_t>(i * nb + p)] = impl->p_x[static_cast<std::size_t>(i)];

      matvec(n, impl->scalar->row_ptr(), impl->scalar->col_ind(), impl->scalar->values(),
          impl->full_x.data(), impl->ax.data());
      std::vector<double> residual(static_cast<std::size_t>(n));
      for (int i = 0; i < n; ++i)
        residual[static_cast<std::size_t>(i)] =
            (B[i] - impl->ax[static_cast<std::size_t>(i)]) * impl->full_scale[static_cast<std::size_t>(i)];
      std::vector<double> dx(static_cast<std::size_t>(n));
      if (rocalution_pair_solve(impl->full, residual.data(), dx.data()) != 0)
        return -1;
      for (int i = 0; i < n; ++i)
        X[i] = impl->full_x[static_cast<std::size_t>(i)] + dx[static_cast<std::size_t>(i)];

      impl->iters = rocalution_pair_iters(impl->pressure) + rocalution_pair_iters(impl->full);
      impl->residual = rocalution_pair_residual(impl->full);
      return 0;
    }

    int linsolv_rocalution_cpr::get_n_iters()
    {
      return impl->iters;
    }

    opendarts::config::mat_float linsolv_rocalution_cpr::get_residual()
    {
      return impl->residual;
    }
  } // namespace linear_solvers
} // namespace opendarts
