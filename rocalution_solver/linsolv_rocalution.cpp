#include "linsolv_rocalution.hpp"

#include <iostream>
#include <vector>

#include <rocalution/rocalution.hpp>

struct linsolv_rocalution::Impl
{
  int device_num = 0;
  int n_rows = 0;
  int nnz = 0;
  int max_iters = 50;
  double tolerance = 1e-10;
  int n_iters = 0;
  double residual = 0.0;
  int verbose = 1;
  bool built = false;

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<double> values;

  rocalution::LocalMatrix<double> A;
  rocalution::LocalVector<double> rhs;
  rocalution::LocalVector<double> x;
  rocalution::FGMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double> fgmres;
  rocalution::Jacobi<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double> jacobi;
};

namespace
{
int g_alive = 0;
bool g_platform = false;

void start_platform(int device_num)
{
  if (g_platform)
    return;
  rocalution::set_device_rocalution(device_num);
  rocalution::init_rocalution();
  rocalution::info_rocalution();
  g_platform = true;
}

void stop_platform()
{
  if (!g_platform)
    return;
  rocalution::stop_rocalution();
  g_platform = false;
}
} // namespace

void rocalution_platform_acquire(int device_num)
{
  start_platform(device_num);
  ++g_alive;
}

void rocalution_platform_release()
{
  if (g_alive > 0)
    --g_alive;
  if (g_alive == 0)
    stop_platform();
}

linsolv_rocalution::linsolv_rocalution(int device_num, int verbose)
  : impl(nullptr)
{
  // rocALUTION objects assert that the backend is already initialized.
  rocalution_platform_acquire(device_num);
  impl = new Impl;
  impl->device_num = device_num;
  impl->verbose = verbose;
}

linsolv_rocalution::~linsolv_rocalution()
{
  if (impl->built)
  {
    impl->fgmres.Clear();
    impl->jacobi.Clear();
    impl->A.Clear();
    impl->rhs.Clear();
    impl->x.Clear();
  }
  delete impl;
  rocalution_platform_release();
}

int linsolv_rocalution::init(int n_rows,
    const int* row_ptr,
    const int* col_ind,
    const double* values,
    int max_iters,
    double tolerance)
{
  if (n_rows < 1 || row_ptr == nullptr || col_ind == nullptr || values == nullptr)
    return -1;

  const int nnz = row_ptr[n_rows];
  if (nnz < n_rows)
    return -1;

  impl->n_rows = n_rows;
  impl->nnz = nnz;
  impl->max_iters = max_iters > 0 ? max_iters : 50;
  impl->tolerance = tolerance > 0.0 ? tolerance : 1e-10;
  impl->row_ptr.assign(row_ptr, row_ptr + n_rows + 1);
  impl->col_ind.assign(col_ind, col_ind + nnz);
  impl->values.assign(values, values + nnz);
  // Leave impl->built alone. setup() must Clear() a live solver before
  // SetOperator; clearing the flag here made the next Newton step skip that.
  return 0;
}

int linsolv_rocalution::setup()
{
  if (impl->n_rows < 1)
    return -1;

  if (impl->built)
  {
    impl->fgmres.Clear();
    impl->jacobi.Clear();
    impl->A.Clear();
    impl->rhs.Clear();
    impl->x.Clear();
    impl->built = false;
  }

  impl->A.CopyFromHostCSR(impl->row_ptr.data(),
      impl->col_ind.data(),
      impl->values.data(),
      "A",
      impl->nnz,
      impl->n_rows,
      impl->n_rows);

  impl->rhs.Allocate("b", impl->n_rows);
  impl->x.Allocate("x", impl->n_rows);

  impl->fgmres.SetPreconditioner(impl->jacobi);
  impl->fgmres.SetOperator(impl->A);
  // Absolute, relative, divergence, maximum iterations.
  // Absolute tolerance stays at 0. FGMRES stops on the true relative residual.
  impl->fgmres.Init(0.0, impl->tolerance, 1.0e8, impl->max_iters);
  const int basis = impl->n_rows < 400 ? impl->n_rows : 400;
  impl->fgmres.SetBasisSize(basis);
  impl->fgmres.Verbose(impl->verbose);
  impl->fgmres.Build();

  impl->A.MoveToAccelerator();
  impl->rhs.MoveToAccelerator();
  impl->x.MoveToAccelerator();
  impl->fgmres.MoveToAccelerator();
  impl->built = true;

  if (impl->verbose)
    std::cout << "rocALUTION: FGMRES + Jacobi on device " << impl->device_num
              << ", n=" << impl->n_rows << ", nnz=" << impl->nnz << std::endl;
  return 0;
}

int linsolv_rocalution::solve(const double* b, double* x)
{
  if (!impl->built || b == nullptr || x == nullptr)
    return -1;

  impl->rhs.CopyFromHostData(b);
  impl->x.Zeros();
  impl->fgmres.Solve(impl->rhs, &impl->x);
  impl->x.CopyToHostData(x);

  impl->n_iters = impl->fgmres.GetIterationCount();
  impl->residual = impl->fgmres.GetCurrentResidual();
  const int status = impl->fgmres.GetSolverStatus();
  if (impl->verbose)
    std::cout << "rocALUTION: status " << status << ", iterations " << impl->n_iters
              << ", residual " << impl->residual << std::endl;

  // 1 = absolute tolerance, 2 = relative tolerance.
  if (status == 1 || status == 2)
    return 0;
  return -1;
}

int linsolv_rocalution::get_n_iters() const
{
  return impl->n_iters;
}

double linsolv_rocalution::get_residual() const
{
  return impl->residual;
}
