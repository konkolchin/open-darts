#include "linsolv_rocalution.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include <rocalution/rocalution.hpp>

// One host system, several rocALUTION solver/preconditioner pairs.
// The matrix is the open-DARTS SuperLU test: 12 rows, offsets -2, 0, +2,
// values -2, 1, 2, right-hand side 1. The reference is that test's CPU solution.
// CG and aggregation AMG are omitted: this matrix is not symmetric positive definite.
// Ruge-Stueben AMG is omitted: rocALUTION rejects this 12-row system as too small.

using Matrix = rocalution::LocalMatrix<double>;
using Vector = rocalution::LocalVector<double>;

static void fill_system(int n, std::vector<int>& row_ptr, std::vector<int>& col_ind, std::vector<double>& values)
{
  row_ptr.assign(n + 1, 0);
  int nnz = 0;
  for (int row = 0; row < n; ++row)
  {
    if (row - 2 >= 0)
      ++nnz;
    ++nnz;
    if (row + 2 < n)
      ++nnz;
    row_ptr[row + 1] = nnz;
  }
  col_ind.resize(nnz);
  values.resize(nnz);
  int k = 0;
  for (int row = 0; row < n; ++row)
  {
    if (row - 2 >= 0)
    {
      col_ind[k] = row - 2;
      values[k] = -2.0;
      ++k;
    }
    col_ind[k] = row;
    values[k] = 1.0;
    ++k;
    if (row + 2 < n)
    {
      col_ind[k] = row + 2;
      values[k] = 2.0;
      ++k;
    }
  }
}

static double max_abs_diff(const std::vector<double>& x, const double* reference, int n)
{
  double error_max = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const double d = std::abs(x[i] - reference[i]);
    if (d > error_max)
      error_max = d;
  }
  return error_max;
}

// Same call sequence as linsolv_rocalution::setup/solve, with the solver and
// preconditioner types supplied by the caller.
template <class Solver, class Prec>
int run_pair(const char* name,
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<double>& b,
    const double* reference,
    double error_tol)
{
  std::cout << "---- " << name << std::endl;

  Matrix A;
  A.CopyFromHostCSR(row_ptr.data(), col_ind.data(), values.data(), name, static_cast<int64_t>(values.size()), n, n);

  Vector rhs;
  Vector x;
  rhs.Allocate("b", n);
  x.Allocate("x", n);

  Solver solver;
  Prec prec;
  solver.SetPreconditioner(prec);
  solver.SetOperator(A);
  solver.Init(1e-10, 1e-10, 1.0e8, 50);
  solver.Verbose(0);
  solver.Build();

  A.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();
  solver.MoveToAccelerator();

  rhs.CopyFromHostData(b.data());
  x.Zeros();
  solver.Solve(rhs, &x);

  std::vector<double> host_x(n);
  x.CopyToHostData(host_x.data());

  const int status = solver.GetSolverStatus();
  const double error_max = max_abs_diff(host_x, reference, n);
  const bool ok = (status == 1 || status == 2) && error_max <= error_tol;
  std::cout << "status " << status
            << ", iterations " << solver.GetIterationCount()
            << ", residual " << solver.GetCurrentResidual()
            << ", max abs diff " << error_max
            << (ok ? " OK" : " FAIL") << std::endl;

  solver.Clear();
  prec.Clear();
  A.Clear();
  rhs.Clear();
  x.Clear();
  return ok ? 0 : 1;
}

int main()
{
  const int n = 12;
  const double error_tol = 1e-8;
  const double reference[12] = {
      -0.071823204419889, -0.071823204419889, 0.535911602209945, 0.535911602209945,
      0.160220994475138, 0.160220994475138, 0.955801104972376, 0.955801104972376,
      0.182320441988950, 0.182320441988950, 1.364640883977901, 1.364640883977901};

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<double> values;
  fill_system(n, row_ptr, col_ind, values);
  const std::vector<double> b(n, 1.0);

  int failed = 0;

  // Keeps the rocALUTION platform initialized for the pairs below.
  linsolv_rocalution adapter(0);
  std::cout << "---- linsolv_rocalution FGMRES + Jacobi" << std::endl;
  std::vector<double> x(n, 0.0);
  int adapter_rc = -1;
  if (adapter.init(n, row_ptr.data(), col_ind.data(), values.data(), 50, 1e-10) == 0
      && adapter.setup() == 0)
    adapter_rc = adapter.solve(b.data(), x.data());
  const double adapter_err = max_abs_diff(x, reference, n);
  for (double value : x)
    std::cout << value << std::endl;
  const bool adapter_ok = adapter_rc == 0 && adapter_err <= error_tol;
  std::cout << "max abs diff " << adapter_err << (adapter_ok ? " OK" : " FAIL") << std::endl;
  if (!adapter_ok)
    ++failed;

  failed += run_pair<rocalution::BiCGStab<Matrix, Vector, double>, rocalution::Jacobi<Matrix, Vector, double>>(
      "BiCGStab + Jacobi", n, row_ptr, col_ind, values, b, reference, error_tol);
  failed += run_pair<rocalution::FGMRES<Matrix, Vector, double>, rocalution::Jacobi<Matrix, Vector, double>>(
      "FGMRES + Jacobi", n, row_ptr, col_ind, values, b, reference, error_tol);
  failed += run_pair<rocalution::GMRES<Matrix, Vector, double>, rocalution::ILU<Matrix, Vector, double>>(
      "GMRES + ILU(0)", n, row_ptr, col_ind, values, b, reference, error_tol);
  failed += run_pair<rocalution::BiCGStab<Matrix, Vector, double>, rocalution::ILU<Matrix, Vector, double>>(
      "BiCGStab + ILU(0)", n, row_ptr, col_ind, values, b, reference, error_tol);
  failed += run_pair<rocalution::GMRES<Matrix, Vector, double>, rocalution::MultiColoredILU<Matrix, Vector, double>>(
      "GMRES + MultiColoredILU", n, row_ptr, col_ind, values, b, reference, error_tol);

  std::cout << (failed == 0 ? "SUITE OK" : "SUITE FAIL") << " failed=" << failed << std::endl;
  return failed == 0 ? 0 : 1;
}
