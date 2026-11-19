#ifndef LINSOLV_ROCALUTION_HPP
#define LINSOLV_ROCALUTION_HPP

// Scalar CSR solver on rocALUTION. Same call order as linsolv_amgx:
// init (structure and values), setup (build the preconditioner), solve (host vectors).
// Block size 1. The Krylov method is GMRES and the preconditioner is Jacobi,
// both of which rocALUTION runs on the GPU.

class linsolv_rocalution
{
public:
  explicit linsolv_rocalution(int device_num = 0, int verbose = 1);
  ~linsolv_rocalution();

  linsolv_rocalution(const linsolv_rocalution&) = delete;
  linsolv_rocalution& operator=(const linsolv_rocalution&) = delete;

  // row_ptr has n_rows + 1 entries. col_ind and values have row_ptr[n_rows] entries.
  // Column indices in each row are increasing.
  int init(int n_rows,
      const int* row_ptr,
      const int* col_ind,
      const double* values,
      int max_iters,
      double tolerance);

  int setup();

  int solve(const double* b, double* x);

  int get_n_iters() const;
  double get_residual() const;

private:
  struct Impl;
  Impl* impl;
};

// Reference-counted rocALUTION backend, shared by every solver in this folder.
void rocalution_platform_acquire(int device_num);
void rocalution_platform_release();

#endif
