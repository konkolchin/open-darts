#ifndef OD_AMGCL_PREC_HPP
#define OD_AMGCL_PREC_HPP

// Host AMGCL preconditioner used from rocALUTION FGMRES.
// The header crosses the HIP translation unit, so it carries no C++ standard
// library types. kind: 1 = ILU(0), 2 = AMG, 3 = CPR (block_size unknowns per cell).

struct amgcl_prec;

amgcl_prec *amgcl_prec_create(int kind,
    int block_size,
    int n_rows,
    const int *row_ptr,
    const int *col_ind,
    const double *values);

void amgcl_prec_apply(const amgcl_prec *prec, const double *rhs, double *x);

void amgcl_prec_destroy(amgcl_prec *prec);

#endif
