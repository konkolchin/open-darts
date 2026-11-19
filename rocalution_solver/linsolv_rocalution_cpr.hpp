//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_ROCALUTION_CPR_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_ROCALUTION_CPR_HPP
//--------------------------------------------------------------------------

// CPR composition: rocALUTION FGMRES, AMGCL CPR preconditioner. Pressure is
// an algebraic multigrid, the full system is ILU(0). Not a change to the two
// standalone rocALUTION suites.

#include "linear_solver.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    class linsolv_rocalution_cpr : public linear_solver
    {
    public:
      linsolv_rocalution_cpr();
      ~linsolv_rocalution_cpr() override;

      int set_prec(opendarts::linear_solvers::linear_solver *prec_input) override;
      int init(opendarts::linear_solvers::csr_matrix_base *A,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance) override;
      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;
      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;
      int reconfigure(const opendarts::linear_solvers::solver_config &config) override;
      int get_n_iters() override;
      opendarts::config::mat_float get_residual() override;

    private:
      struct Impl;
      Impl *impl;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_ROCALUTION_CPR_HPP
//--------------------------------------------------------------------------
