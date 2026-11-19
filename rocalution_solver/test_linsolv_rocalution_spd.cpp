#include <cmath>
#include <iostream>
#include <vector>

#include <rocalution/rocalution.hpp>

// Second suite. The stencil is the one in
// tests/cpp/unit/linear_solvers/linsolv_hypre_amg.cpp:
// generate_tridiagonal_matrix(A, n, constant_filled_block, is_SPD=true)
// places +1, -2, +1 on offsets -2, 0, +2. That matrix is symmetric but
// negative definite, and that test uses n = 120.
// rocALUTION AMG (base_amg.cpp) stops when the row count is not above its
// coarse-size limit of 300. This file keeps the same offsets, uses the
// positive Laplacian (+2 on the diagonal, -1 on ±2), and sets n = 400.
// The right-hand side is 1, as in the hypre test.
// The check is the same idea as that test: ||b - Ax|| / ||b||.

using Matrix = rocalution::LocalMatrix<double>;
using Vector = rocalution::LocalVector<double>;

static void fill_spd_laplacian(int n, std::vector<int>& row_ptr, std::vector<int>& col_ind, std::vector<double>& values)
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
      values[k] = -1.0;
      ++k;
    }
    col_ind[k] = row;
    values[k] = 2.0;
    ++k;
    if (row + 2 < n)
    {
      col_ind[k] = row + 2;
      values[k] = -1.0;
      ++k;
    }
  }
}

static double relative_residual(int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<double>& b,
    const std::vector<double>& x)
{
  double resid2 = 0.0;
  double b2 = 0.0;
  for (int row = 0; row < n; ++row)
  {
    double ax = 0.0;
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
      ax += values[k] * x[col_ind[k]];
    const double r = b[row] - ax;
    resid2 += r * r;
    b2 += b[row] * b[row];
  }
  return std::sqrt(resid2 / b2);
}

template <class Solver, class Prec>
int run_pair(const char* name,
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<double>& values,
    const std::vector<double>& b,
    double error_tol,
    bool on_accelerator)
{
  std::cout << "---- " << name << (on_accelerator ? "" : " (host)") << std::endl;

  Matrix A;
  A.CopyFromHostCSR(row_ptr.data(), col_ind.data(), values.data(), name,
      static_cast<int64_t>(values.size()), n, n);

  Vector rhs;
  Vector sol;
  rhs.Allocate("b", n);
  sol.Allocate("x", n);

  Solver solver;
  Prec prec;
  solver.SetPreconditioner(prec);
  solver.SetOperator(A);
  solver.Init(1e-10, 1e-10, 1.0e8, 1000);
  solver.Verbose(0);
  solver.Build();

  // IC's triangular solve in this rocALUTION build asserts unless the factor,
  // the inverse diagonal, and both vectors are on one backend. Moving that
  // pair to the GPU leaves them split, so IC stays on the host.
  if (on_accelerator)
  {
    A.MoveToAccelerator();
    rhs.MoveToAccelerator();
    sol.MoveToAccelerator();
    solver.MoveToAccelerator();
  }

  rhs.CopyFromHostData(b.data());
  sol.Zeros();
  solver.Solve(rhs, &sol);

  std::vector<double> x(n);
  sol.CopyToHostData(x.data());
  const double rel = relative_residual(n, row_ptr, col_ind, values, b, x);
  const int status = solver.GetSolverStatus();
  const bool ok = (status == 1 || status == 2) && rel <= error_tol;
  std::cout << "status " << status
            << ", iterations " << solver.GetIterationCount()
            << ", residual " << solver.GetCurrentResidual()
            << ", ||b-Ax||/||b|| " << rel
            << (ok ? " OK" : " FAIL") << std::endl;

  solver.Clear();
  prec.Clear();
  A.Clear();
  rhs.Clear();
  sol.Clear();
  return ok ? 0 : 1;
}

int main()
{
  const int n = 400;
  const double error_tol = 1e-8;

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<double> values;
  fill_spd_laplacian(n, row_ptr, col_ind, values);
  const std::vector<double> b(n, 1.0);

  rocalution::set_device_rocalution(0);
  rocalution::init_rocalution();
  rocalution::info_rocalution();

  int failed = 0;
  failed += run_pair<rocalution::CG<Matrix, Vector, double>, rocalution::Jacobi<Matrix, Vector, double>>(
      "CG + Jacobi", n, row_ptr, col_ind, values, b, error_tol, true);
  failed += run_pair<rocalution::CG<Matrix, Vector, double>, rocalution::IC<Matrix, Vector, double>>(
      "CG + IC(0)", n, row_ptr, col_ind, values, b, error_tol, false);
  failed += run_pair<rocalution::CG<Matrix, Vector, double>, rocalution::SAAMG<Matrix, Vector, double>>(
      "CG + SAAMG", n, row_ptr, col_ind, values, b, error_tol, true);
  failed += run_pair<rocalution::CG<Matrix, Vector, double>, rocalution::UAAMG<Matrix, Vector, double>>(
      "CG + UAAMG", n, row_ptr, col_ind, values, b, error_tol, true);
  failed += run_pair<rocalution::CG<Matrix, Vector, double>, rocalution::RugeStuebenAMG<Matrix, Vector, double>>(
      "CG + RugeStuebenAMG", n, row_ptr, col_ind, values, b, error_tol, true);

  rocalution::stop_rocalution();
  std::cout << (failed == 0 ? "SUITE OK" : "SUITE FAIL") << " failed=" << failed << std::endl;
  return failed == 0 ? 0 : 1;
}
