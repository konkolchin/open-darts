#include "amgcl_prec.hpp"

#include <memory>
#include <tuple>
#include <vector>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/spai0.hpp>

namespace
{
using backend = amgcl::backend::builtin<double>;
using pressure_amg =
    amgcl::amg<backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;
using ilu0 = amgcl::relaxation::as_preconditioner<backend, amgcl::relaxation::ilu0>;
using cpr = amgcl::preconditioner::cpr<pressure_amg, ilu0>;

struct apply_base
{
  virtual ~apply_base() = default;
  virtual void apply(const std::vector<double> &rhs, std::vector<double> &x) const = 0;
};

template <class Prec>
struct holder : apply_base
{
  Prec prec;

  template <class Matrix>
  explicit holder(const Matrix &A)
    : prec(A)
  {
  }

  template <class Matrix, class Params>
  holder(const Matrix &A, const Params &prm)
    : prec(A, prm)
  {
  }

  void apply(const std::vector<double> &rhs, std::vector<double> &x) const override
  {
    prec.apply(rhs, x);
  }
};
} // namespace

struct amgcl_prec
{
  std::unique_ptr<apply_base> op;
  int n = 0;
};

amgcl_prec *amgcl_prec_create(int kind,
    int block_size,
    int n_rows,
    const int *row_ptr,
    const int *col_ind,
    const double *values)
{
  if (n_rows < 1 || row_ptr == nullptr || col_ind == nullptr || values == nullptr)
    return nullptr;
  const int nnz = row_ptr[n_rows];
  if (nnz < n_rows)
    return nullptr;

  // AMGCL copies the matrix during construction. The tie only has to live
  // for that call, and int index arrays are what crs_tuple accepts.
  const int n = n_rows;
  std::vector<int> ptr(row_ptr, row_ptr + n_rows + 1);
  std::vector<int> col(col_ind, col_ind + nnz);
  std::vector<double> val(values, values + nnz);
  const auto matrix = std::tie(n, ptr, col, val);

  auto *prec = new amgcl_prec;
  prec->n = n_rows;
  const bool cpr_shape = kind == 3 && block_size > 1 && n_rows % block_size == 0;
  if (cpr_shape)
  {
    cpr::params prm;
    prm.block_size = block_size;
    prec->op = std::make_unique<holder<cpr>>(matrix, prm);
  }
  else if (kind == 2)
    prec->op = std::make_unique<holder<pressure_amg>>(matrix);
  else
    prec->op = std::make_unique<holder<ilu0>>(matrix);
  return prec;
}

void amgcl_prec_apply(const amgcl_prec *prec, const double *rhs, double *x)
{
  if (prec == nullptr || prec->op == nullptr || rhs == nullptr || x == nullptr)
    return;
  std::vector<double> b(rhs, rhs + prec->n);
  std::vector<double> y(static_cast<std::size_t>(prec->n), 0.0);
  prec->op->apply(b, y);
  for (int i = 0; i < prec->n; ++i)
    x[i] = y[static_cast<std::size_t>(i)];
}

void amgcl_prec_destroy(amgcl_prec *prec)
{
  delete prec;
}
