#ifndef LINSOLV_ROCALUTION_BACKEND_HPP
#define LINSOLV_ROCALUTION_BACKEND_HPP

// Pointer-only rocALUTION pairs. The header is safe for an ABI-0 caller.
// kind 0: GMRES + SAAMG, left on the host (AMG setup stays on the host).
// kind 1: GMRES + ILU(0), moved to the accelerator.

struct rocalution_pair;

rocalution_pair *rocalution_pair_create(int kind, int device_num);
void rocalution_pair_destroy(rocalution_pair *pair);

int rocalution_pair_setup(rocalution_pair *pair,
    int n_rows,
    const int *row_ptr,
    const int *col_ind,
    const double *values,
    int max_iters,
    double tolerance);

// Returns 0 when rocALUTION reports an absolute or relative tolerance.
int rocalution_pair_solve(rocalution_pair *pair, const double *b, double *x);

int rocalution_pair_iters(const rocalution_pair *pair);
double rocalution_pair_residual(const rocalution_pair *pair);

#endif
