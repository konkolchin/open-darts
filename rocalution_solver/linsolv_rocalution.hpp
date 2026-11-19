#ifndef LINSOLV_ROCALUTION_HPP
#define LINSOLV_ROCALUTION_HPP

// Scalar CSR solver. rocALUTION FGMRES is the Krylov method. The preconditioner
// is either rocALUTION Jacobi (the regression tests) or an AMGCL preconditioner
// (ILU(0), AMG, or CPR) applied through rocALUTION's preconditioner slot.
// FGMRES is right-preconditioned, so it stops on the original residual.

class linsolv_rocalution
{
public:
  explicit linsolv_rocalution(int device_num = 0, int verbose = 1);
  ~linsolv_rocalution();

  linsolv_rocalution(const linsolv_rocalution&) = delete;
  linsolv_rocalution& operator=(const linsolv_rocalution&) = delete;

  // kind 0: rocALUTION Jacobi. 1: AMGCL ILU(0). 2: AMGCL AMG. 3: AMGCL CPR.
  // block_size is the number of unknowns per cell and is used by CPR only.
  void set_preconditioner(int kind, int block_size);

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
