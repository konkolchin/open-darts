#include "linsolv_rocalution.hpp"

#include <cmath>
#include <iostream>
#include <vector>

// Same 12-unknown system as tests/cpp/unit/linear_solvers/linsolv_superlu.cpp
// and linsolv_amgx.cpp: block size 1, diagonal blocks, rhs = 1.
// Offsets -2, 0, +2 carry -2, 1, 2.

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

  linsolv_rocalution solver(0);
  if (solver.init(n, row_ptr.data(), col_ind.data(), values.data(), /*max_iters=*/50, /*tolerance=*/1e-10) != 0)
  {
    std::cout << "FAIL init" << std::endl;
    return 1;
  }
  if (solver.setup() != 0)
  {
    std::cout << "FAIL setup" << std::endl;
    return 1;
  }

  std::vector<double> b(n, 1.0);
  std::vector<double> x(n, 0.0);
  if (solver.solve(b.data(), x.data()) != 0)
  {
    std::cout << "FAIL solve" << std::endl;
    return 1;
  }

  double error_max = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const double d = std::abs(x[i] - reference[i]);
    if (d > error_max)
      error_max = d;
    std::cout << x[i] << std::endl;
  }
  std::cout << "max abs diff " << error_max << std::endl;
  if (error_max > error_tol)
  {
    std::cout << "FAIL" << std::endl;
    return 1;
  }
  std::cout << "OK" << std::endl;
  return 0;
}
