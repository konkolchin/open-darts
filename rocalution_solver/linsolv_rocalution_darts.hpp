//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_ROCALUTION_DARTS_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_ROCALUTION_DARTS_HPP
//--------------------------------------------------------------------------

// DARTS linear_solver that expands a host block-CSR Jacobian to scalar CSR
// and solves it with the existing rocALUTION GMRES + Jacobi adapter.

#include <memory>

#include "linear_solver.hpp"
#include "linsolv_rocalution.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    class block_csr_matrix;
    class scalar_csr_adapter;

    class linsolv_rocalution_solver : public linear_solver
    {
    public:
      linsolv_rocalution_solver();
      ~linsolv_rocalution_solver() override;

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
      int handoff();

      block_csr_matrix *A_ = nullptr;
      std::unique_ptr<scalar_csr_adapter> scalar_;
      linsolv_rocalution solver_;
      int max_iters_ = 50;
      double tolerance_ = 1e-6;
      bool ready_ = false;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_ROCALUTION_DARTS_HPP
//--------------------------------------------------------------------------
