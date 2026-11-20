#include "amgcl_prec.hpp"

#ifndef __HIPCC__
#error "AMGCL on this port is the HIP GPU backend. Compile amgcl_prec.cpp with hipcc."
#endif

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <hipsparse/hipsparse.h>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/hip.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/rocsparse_ilu0.hpp>
#include <amgcl/relaxation/spai0.hpp>

namespace
{
using backend = amgcl::backend::hip<double>;
using pressure_amg =
    amgcl::amg<backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;
using ilu0 = amgcl::relaxation::as_preconditioner<backend, amgcl::relaxation::ilu0>;
using cpr = amgcl::preconditioner::cpr<pressure_amg, ilu0>;
using device_vector = thrust::device_vector<double>;

struct apply_base
{
  virtual ~apply_base() = default;
  virtual void apply(const device_vector &rhs, device_vector &x) const = 0;
};

template <class Prec>
struct holder : apply_base
{
  Prec prec;

  template <class Matrix>
  holder(const Matrix &A, const backend::params &bprm)
    : prec(A, typename Prec::params(), bprm)
  {
  }

  template <class Matrix, class Params>
  holder(const Matrix &A, const Params &prm, const backend::params &bprm)
    : prec(A, prm, bprm)
  {
  }

  void apply(const device_vector &rhs, device_vector &x) const override
  {
    prec.apply(rhs, x);
  }
};
} // namespace

struct amgcl_prec
{
  // Destroyed after op: the HIP preconditioner uses this handle.
  hipsparseHandle_t handle = nullptr;
  std::unique_ptr<apply_base> op;
  device_vector rhs;
  device_vector x;
  int n = 0;

  ~amgcl_prec()
  {
    op.reset();
    if (handle != nullptr)
      hipsparseDestroy(handle);
  }
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

  auto *prec = new amgcl_prec;
  if (hipsparseCreate(&prec->handle) != HIPSPARSE_STATUS_SUCCESS)
  {
    delete prec;
    return nullptr;
  }
  hipsparseSetPointerMode(prec->handle, HIPSPARSE_POINTER_MODE_HOST);

  const int n = n_rows;
  std::vector<int> ptr(row_ptr, row_ptr + n_rows + 1);
  std::vector<int> col(col_ind, col_ind + nnz);
  std::vector<double> val(values, values + nnz);
  const auto matrix = std::tie(n, ptr, col, val);
  const backend::params bprm(prec->handle);

  try
  {
    prec->n = n_rows;
    const bool cpr_shape = kind == 3 && block_size > 1 && n_rows % block_size == 0;
    if (cpr_shape)
    {
      cpr::params prm;
      prm.block_size = block_size;
      prec->op = std::make_unique<holder<cpr>>(matrix, prm, bprm);
    }
    else if (kind == 2)
      prec->op = std::make_unique<holder<pressure_amg>>(matrix, bprm);
    else
      prec->op = std::make_unique<holder<ilu0>>(matrix, bprm);
  }
  catch (const std::exception &ex)
  {
    std::cerr << "AMGCL HIP preconditioner: " << ex.what() << std::endl;
    delete prec;
    return nullptr;
  }
  return prec;
}

void amgcl_prec_apply(amgcl_prec *prec, const double *rhs, double *x)
{
  if (prec == nullptr || prec->op == nullptr || rhs == nullptr || x == nullptr)
    return;
  if (static_cast<int>(prec->rhs.size()) != prec->n)
    prec->rhs.resize(static_cast<std::size_t>(prec->n));
  if (static_cast<int>(prec->x.size()) != prec->n)
    prec->x.resize(static_cast<std::size_t>(prec->n));
  thrust::copy_n(thrust::device_pointer_cast(rhs), prec->n, prec->rhs.begin());
  prec->op->apply(prec->rhs, prec->x);
  thrust::copy_n(prec->x.begin(), prec->n, thrust::device_pointer_cast(x));
}

void amgcl_prec_destroy(amgcl_prec *prec)
{
  delete prec;
}
