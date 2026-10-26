//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//    Netherlands eScience Center
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_BLOCK_ROW_PERM_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_BLOCK_ROW_PERM_HPP
//--------------------------------------------------------------------------

#include <vector>

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Block-local EQUATION (row) reordering wrapper: solves P A x = P b with the
        inner chain, where P permutes the N local equations inside every block row.
        A left permutation does not change x, so nothing is transformed back.

        WHY. DARTS orders the local equations as the component balances 0..nc-1
        and the local unknowns as [p, z_0, .., z_{nc-2}], so local equation k sits
        on the diagonal of unknown k-1 and equation 0 sits on PRESSURE. For flowing
        components that pairing is harmless: every balance depends strongly on p.
        With a NON-FLOWING first component (a mineral in a reactive-transport
        model) the (0,0) entry of every diagonal block is d(mineral balance)/dp,
        which goes to zero as the mineral is exhausted. cuSPARSE bsrilu02 (and any
        unpivoted block ILU) eliminates diagonal blocks WITHOUT pivoting, so it
        divides by that entry.

        The wrapper owns a device copy of the block values (same sparsity) and
        permutes the rows of every stored block at setup(); solve() permutes the
        right-hand side. Cost: one O(nnz N^2) copy kernel per setup.

        @tparam N_BLOCK_SIZE block size N.
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_block_row_perm : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                                   public opendarts::linear_solvers::linear_solver_base
    {
    public:
      /** @param on_device the matrix values / vectors handed to setup()/solve()
              are device pointers (GPU engine); otherwise host.
          @param perm size N; perm[r] = ORIGINAL local equation that becomes local
              row r. Must be a permutation of 0..N-1.
          @param n_perm_rows number of LEADING block rows to permute; block rows
              at or beyond it are copied unchanged. 0 = all block rows.
          @param skip_rows block rows (below n_perm_rows) copied unchanged as
              well. The GPU engine permutes every block row except the well
              HEADS, whose rows hold control and state-copy equations rather than
              component balances; well BODY rows carry the same balances as the
              reservoir cells and are permuted with them, so that a wrapped
              Schur elimination finds its eliminated equation in the same local
              row everywhere. */
      linsolv_block_row_perm(bool on_device, const std::vector<int> &perm,
                             opendarts::config::index_t n_perm_rows = 0,
                             const std::vector<int> &skip_rows = {});

      ~linsolv_block_row_perm();

      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      //////////////////////
      // linear_solver_base
      //////////////////////

      int solve(opendarts::linear_solvers::csr_matrix_base * /*matrix*/,
        opendarts::config::mat_float *v, opendarts::config::mat_float *r) override
      {
        return solve(v, r);
      }

      int init(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;

      //////////////////////
      // linsolv_iface
      //////////////////////

      /// Inner solver on the row-permuted system. Set BEFORE init(). Ownership
      /// follows linsolv_schur_elim: not owned unless set_inner_owned(true).
      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override
      {
        inner = prec_input;
        return 0;
      }

      void set_inner_owned(bool owned) { inner_owned = owned; }

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input, int max_iters, double tolerance) override
      {
        return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input),
            static_cast<opendarts::config::index_t>(max_iters), static_cast<opendarts::config::mat_float>(tolerance));
      }
      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input));
      }

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override { return inner ? inner->get_n_iters() : 0; }
      opendarts::config::mat_float get_residual() override { return inner ? inner->get_residual() : 0; }
      void set_last_outer_iters(int n_iters) override { if (inner) inner->set_last_outer_iters(n_iters); }

    private:
      void free_device();

      opendarts::linear_solvers::linsolv_iface *inner = nullptr;
      bool inner_owned = false;
      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *permuted = nullptr;
      opendarts::linear_solvers::csr_matrix_base *A_saved = nullptr;

      bool on_device = false;
      std::vector<uint8_t> perm;               // size N
      opendarts::config::index_t n_perm_rows_requested = 0;
      opendarts::config::index_t n_perm_rows = 0;      // block rows actually permuted
      std::vector<int> skip_rows;                      // left unchanged below n_perm_rows_requested
      std::vector<uint8_t> permute_mask;               // per block row, host copy
      uint8_t *permute_mask_d = nullptr;
      opendarts::config::index_t n_rows = 0;
      opendarts::config::index_t n_nnz = 0;

      std::vector<opendarts::config::mat_float> b_perm;   // host path
      uint8_t *perm_d = nullptr;
      opendarts::config::mat_float *b_perm_d = nullptr;
    };

  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif /* OPENDARTS_LINEAR_SOLVERS_LINSOLV_BLOCK_ROW_PERM_HPP */
//--------------------------------------------------------------------------
