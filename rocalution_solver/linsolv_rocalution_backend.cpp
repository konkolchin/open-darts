#include "linsolv_rocalution_backend.hpp"
#include "linsolv_rocalution.hpp"

#include <vector>

#include <rocalution/rocalution.hpp>

struct rocalution_pair
{
  virtual ~rocalution_pair() = default;
  virtual int setup(int n, const int *row, const int *col, const double *val,
      int max_iters, double tol) = 0;
  virtual int solve(const double *b, double *x) = 0;
  int iters = 0;
  double residual = 0.0;
};

using Matrix = rocalution::LocalMatrix<double>;
using Vector = rocalution::LocalVector<double>;

namespace
{

template <class Prec, bool on_accelerator>
struct pair_impl : rocalution_pair
{
  Matrix A;
  Vector rhs;
  Vector x;
  rocalution::GMRES<Matrix, Vector, double> solver;
  Prec prec;
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<double> values;
  int n_rows = 0;
  bool built = false;

  ~pair_impl() override
  {
    clear();
  }

  void clear()
  {
    if (!built)
      return;
    solver.Clear();
    prec.Clear();
    A.Clear();
    rhs.Clear();
    x.Clear();
    built = false;
  }

  int setup(int n, const int *row, const int *col, const double *val,
      int max_iters, double tol) override
  {
    if (n < 1 || row == nullptr || col == nullptr || val == nullptr)
      return -1;
    const int nnz = row[n];
    if (nnz < n)
      return -1;
    clear();
    n_rows = n;
    row_ptr.assign(row, row + n + 1);
    col_ind.assign(col, col + nnz);
    values.assign(val, val + nnz);

    A.CopyFromHostCSR(row_ptr.data(), col_ind.data(), values.data(), "A",
        static_cast<int64_t>(nnz), n, n);
    rhs.Allocate("b", n);
    x.Allocate("x", n);
    solver.SetPreconditioner(prec);
    solver.SetOperator(A);
    const int budget = max_iters > 0 ? max_iters : 50;
    const double tolerance = tol > 0.0 ? tol : 1e-8;
    solver.Init(0.0, tolerance, 1.0e8, budget);
    const int basis = n < 400 ? n : 400;
    solver.SetBasisSize(basis);
    solver.Verbose(0);
    solver.Build();
    if (on_accelerator)
    {
      A.MoveToAccelerator();
      rhs.MoveToAccelerator();
      x.MoveToAccelerator();
      solver.MoveToAccelerator();
    }
    built = true;
    return 0;
  }

  int solve(const double *b, double *host_x) override
  {
    if (!built || b == nullptr || host_x == nullptr)
      return -1;
    rhs.CopyFromHostData(b);
    x.Zeros();
    solver.Solve(rhs, &x);
    x.CopyToHostData(host_x);
    iters = solver.GetIterationCount();
    residual = solver.GetCurrentResidual();
    const int status = solver.GetSolverStatus();
    if (status == 1 || status == 2)
      return 0;
    return -1;
  }
};
} // namespace

rocalution_pair *rocalution_pair_create(int kind, int device_num)
{
  rocalution_platform_acquire(device_num);
  if (kind == 0)
    return new pair_impl<rocalution::SAAMG<Matrix, Vector, double>, false>();
  return new pair_impl<rocalution::ILU<Matrix, Vector, double>, true>();
}

void rocalution_pair_destroy(rocalution_pair *pair)
{
  delete pair;
  rocalution_platform_release();
}

int rocalution_pair_setup(rocalution_pair *pair,
    int n_rows,
    const int *row_ptr,
    const int *col_ind,
    const double *values,
    int max_iters,
    double tolerance)
{
  if (pair == nullptr)
    return -1;
  return pair->setup(n_rows, row_ptr, col_ind, values, max_iters, tolerance);
}

int rocalution_pair_solve(rocalution_pair *pair, const double *b, double *x)
{
  if (pair == nullptr)
    return -1;
  return pair->solve(b, x);
}

int rocalution_pair_iters(const rocalution_pair *pair)
{
  return pair == nullptr ? 0 : pair->iters;
}

double rocalution_pair_residual(const rocalution_pair *pair)
{
  return pair == nullptr ? 0.0 : pair->residual;
}
