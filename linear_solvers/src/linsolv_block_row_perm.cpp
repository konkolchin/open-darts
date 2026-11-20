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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_GPU
#include "od_gpu_runtime.hpp"
#endif
#ifdef OD_PORT_BENCH
#include "port_bench_time.hpp"
#endif

#include "linsolv_block_row_perm.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      using opendarts::config::index_t;
      using opendarts::config::mat_float;

#if defined(WITH_GPU) && (defined(__CUDACC__) || defined(__HIPCC__))
      constexpr int BRP_BLOCK = 256;

      // dst block e, local row r  <-  src block e, local row perm[r]; block rows
      // with mask 0 are copied unchanged.
      template <uint8_t N>
      __global__ void brp_values_kernel(index_t n_rows, const uint8_t *mask, const index_t *rows_ptr,
        const uint8_t *perm, const mat_float *src, mat_float *dst)
      {
        const index_t i = (index_t)(blockIdx.x * blockDim.x + threadIdx.x);
        if (i >= n_rows)
          return;
        const bool permute = mask[i];
        for (index_t e = rows_ptr[i]; e < rows_ptr[i + 1]; ++e)
        {
          const mat_float *s = src + (size_t)e * N * N;
          mat_float *d = dst + (size_t)e * N * N;
          for (uint8_t r = 0; r < N; ++r)
          {
            const uint8_t sr = permute ? perm[r] : r;
            for (uint8_t c = 0; c < N; ++c)
              d[r * N + c] = s[sr * N + c];
          }
        }
      }

      template <uint8_t N>
      __global__ void brp_rhs_kernel(index_t n_rows, const uint8_t *mask, const uint8_t *perm,
        const mat_float *src, mat_float *dst)
      {
        const index_t i = (index_t)(blockIdx.x * blockDim.x + threadIdx.x);
        if (i >= n_rows)
          return;
        const bool permute = mask[i];
        for (uint8_t r = 0; r < N; ++r)
          dst[(size_t)i * N + r] = src[(size_t)i * N + (permute ? perm[r] : r)];
      }
#endif
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_block_row_perm<N_BLOCK_SIZE>::linsolv_block_row_perm(bool on_device_,
      const std::vector<int> &perm_, index_t n_perm_rows_, const std::vector<int> &skip_rows_)
      : on_device(on_device_), n_perm_rows_requested(n_perm_rows_), skip_rows(skip_rows_)
    {
      constexpr int N = N_BLOCK_SIZE;
      if ((int)perm_.size() != N)
        throw std::invalid_argument("linsolv_block_row_perm: perm must have exactly N = " +
            std::to_string(N) + " entries, got " + std::to_string(perm_.size()));
      std::vector<bool> seen(N, false);
      for (int p : perm_)
      {
        if (p < 0 || p >= N || seen[p])
          throw std::invalid_argument("linsolv_block_row_perm: perm is not a permutation of 0..N-1");
        seen[p] = true;
      }
      perm.assign(perm_.begin(), perm_.end());
#ifndef WITH_GPU
      if (on_device)
        throw std::invalid_argument("linsolv_block_row_perm: on_device requested in a build without GPU support");
#endif
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_block_row_perm<N_BLOCK_SIZE>::~linsolv_block_row_perm()
    {
      // ORDER MATTERS: the inner chain borrows `permuted` (linsolv_cusparse_ilu
      // keeps its cuSPARSE handle, CPR/GMRES its device arrays), so it goes first.
      if (inner_owned)
        delete inner;
      inner = nullptr;
      free_device();
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_block_row_perm<N_BLOCK_SIZE>::free_device()
    {
#ifdef WITH_GPU
      if (perm_d) { cudaFree(perm_d); perm_d = nullptr; }
      if (permute_mask_d) { cudaFree(permute_mask_d); permute_mask_d = nullptr; }
      if (b_perm_d) { cudaFree(b_perm_d); b_perm_d = nullptr; }
      // csr_matrix<N> has no destructor: its device arrays, work vectors and
      // cuSPARSE handle are only released by an explicit free_device().
      if (permuted && on_device)
        permuted->free_device();
#endif
      delete permuted;
      permuted = nullptr;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_block_row_perm<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A,
      index_t max_iters, mat_float tolerance)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      if (!inner)
      {
        printf("linsolv_block_row_perm: no inner solver attached (set_prec) before init\n");
        return 1;
      }
      A_saved = A;
      const index_t *rows_ptr = A->get_rows_ptr();
      // Re-init (model.reset(), the adjoint driver): an unchanged sparsity keeps
      // the existing permuted matrix -- the inner chain holds pointers into it,
      // and csr_matrix device storage is expensive to cycle. Same policy as the
      // pressure matrix of linsolv_cpr_gpu.
      if (permuted && n_rows == A->n_rows && n_nnz == rows_ptr[A->n_rows])
      {
        inner->init_timer_nodes(this->timer_setup, this->timer_solve);
        return inner->init(permuted, max_iters, tolerance);
      }
      free_device();

      n_rows = A->n_rows;
      n_nnz = rows_ptr[n_rows];
      // which block rows get the permutation: the leading n_perm_rows_requested
      // (0 = all) minus the explicitly skipped ones
      const index_t n_leading = (n_perm_rows_requested <= 0 || n_perm_rows_requested > n_rows)
          ? n_rows : n_perm_rows_requested;
      permute_mask.assign((size_t)n_rows, 0);
      std::fill(permute_mask.begin(), permute_mask.begin() + n_leading, 1);
      for (int row : skip_rows)
        if (row >= 0 && row < n_rows)
          permute_mask[row] = 0;
      n_perm_rows = (index_t)std::count(permute_mask.begin(), permute_mask.end(), (uint8_t)1);

      permuted = new opendarts::linear_solvers::csr_matrix<N>;
      permuted->init(n_rows, n_rows, n_nnz);
      std::memcpy(permuted->rows_ptr.data(), rows_ptr, sizeof(index_t) * (n_rows + 1));
      std::memcpy(permuted->cols_ind.data(), A->get_cols_ind(), sizeof(index_t) * n_nnz);
      permuted->diag_ind.resize(n_rows);
      std::memcpy(permuted->diag_ind.data(), A->get_diag_ind(), sizeof(index_t) * n_rows);

#ifdef WITH_GPU
      if (on_device)
      {
        // A failed allocation must stop HERE: continuing would let the first
        // setup() kernel write through a null pointer and surface later as a
        // sticky CUDA error inside an unrelated call.
        if (permuted->init_device(n_rows, n_nnz) != 0 || permuted->copy_struct_to_device() != 0 ||
            cudaMalloc((void **)&perm_d, N) != cudaSuccess ||
            cudaMalloc((void **)&permute_mask_d, (size_t)n_rows) != cudaSuccess ||
            cudaMalloc((void **)&b_perm_d, sizeof(mat_float) * (size_t)n_rows * N) != cudaSuccess ||
            cudaMemcpy(perm_d, perm.data(), N, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(permute_mask_d, permute_mask.data(), (size_t)n_rows, cudaMemcpyHostToDevice) != cudaSuccess)
        {
          printf("linsolv_block_row_perm: device allocation failed (%.2f GB for the permuted values)\n",
              (double)n_nnz * N * N * sizeof(mat_float) / 1e9);
          free_device();
          return 1;
        }
      }
      else
#endif
        b_perm.assign((size_t)n_rows * N, 0);

      std::string order;
      for (uint8_t r = 0; r < N; ++r)
        order += (r ? "," : "") + std::to_string((int)perm[r]);
      printf("linsolv_block_row_perm<%d>: local equation order [%s] applied to %d of %d block rows\n",
          (int)N, order.c_str(), (int)n_perm_rows, (int)n_rows);

      inner->init_timer_nodes(this->timer_setup, this->timer_solve);
      return inner->init(permuted, max_iters, tolerance);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_block_row_perm<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      if (!permuted || !inner)
      {
        printf("linsolv_block_row_perm: setup() before init()\n");
        return 1;
      }
      A_saved = A;
      // declared outside the GPU guard: the host path below times itself too
      ::timer_node *t = this->timer_setup ? &this->timer_setup->node["BLOCK_ROW_PERM"] : nullptr;
#ifdef WITH_GPU
      if (on_device)
      {
#if defined(__CUDACC__) || defined(__HIPCC__)
        const int grid = (int)((n_rows + BRP_BLOCK - 1) / BRP_BLOCK);
        if (t)
          t->start();
        brp_values_kernel<N><<<grid, BRP_BLOCK>>>(n_rows, permute_mask_d, A->get_rows_ptr_d(), perm_d,
            A->get_values_d(), permuted->values_d);
        const cudaError_t launch = cudaGetLastError();
        if (t)
          t->stop();   // launch only, unless DARTS_TIMER_SYNC=1 makes the timer synchronise
        if (launch != cudaSuccess)
        {
          printf("linsolv_block_row_perm: values kernel launch failed\n");
          return 1;
        }
#else
        printf("linsolv_block_row_perm: device path needs a CUDA or HIP translation unit\n");
        return 1;
#endif
        return inner->setup(permuted);
      }
#endif
      const index_t *rows_ptr = A->get_rows_ptr();
      const mat_float *src = A->get_values();
      mat_float *dst = permuted->values.data();
      if (t)
        t->start();
      for (index_t i = 0; i < n_rows; ++i)
      {
        const bool permute = permute_mask[(size_t)i];
        for (index_t e = rows_ptr[i]; e < rows_ptr[i + 1]; ++e)
          for (uint8_t r = 0; r < N; ++r)
            std::memcpy(dst + ((size_t)e * N + r) * N,
                        src + ((size_t)e * N + (permute ? perm[r] : r)) * N, sizeof(mat_float) * N);
      }
      if (t)
        t->stop();
      return inner->setup(permuted);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_block_row_perm<N_BLOCK_SIZE>::solve(mat_float *B, mat_float *X)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      ::timer_node *t = this->timer_solve ? &this->timer_solve->node["BLOCK_ROW_PERM"] : nullptr;
#ifdef WITH_GPU
      if (on_device)
      {
#if defined(__CUDACC__) || defined(__HIPCC__)
        const int grid = (int)((n_rows + BRP_BLOCK - 1) / BRP_BLOCK);
        if (t)
          t->start();
        brp_rhs_kernel<N><<<grid, BRP_BLOCK>>>(n_rows, permute_mask_d, perm_d, B, b_perm_d);
        const cudaError_t launch = cudaGetLastError();
        if (t)
          t->stop();
        if (launch != cudaSuccess)
        {
          printf("linsolv_block_row_perm: rhs kernel launch failed\n");
          return -1;
        }
#endif
        // P A x = P b has the SAME solution x: nothing to transform back, and the
        // caller's initial guess in X is passed straight through.
        return inner->solve(b_perm_d, X);
      }
#endif
      if (t)
        t->start();
      for (index_t i = 0; i < n_rows; ++i)
      {
        const bool permute = permute_mask[(size_t)i];
        for (uint8_t r = 0; r < N; ++r)
          b_perm[(size_t)i * N + r] = B[(size_t)i * N + (permute ? perm[r] : r)];
      }
      if (t)
        t->stop();
      return inner->solve(b_perm.data(), X);
    }

    // Explicit template instantiations are generated by CMake via
    // od_emit_template_instantiations(); see solvers/src/CMakeLists.txt.
    // Edit the block-size range there, not here.
#ifndef OD_PORT_BENCH
#include "linsolv_block_row_perm_instantiations.inc"
#endif

#ifdef OD_PORT_BENCH
// Swap the two local rows of every 2x2 block. One stored block per row.
extern "C" int od_bench_brp(int n, int warmup, int samples, double *median_ms, double *checksum)
{
  using opendarts::config::index_t;
  using opendarts::config::mat_float;
  index_t *rows = nullptr;
  uint8_t *mask = nullptr;
  uint8_t *perm = nullptr;
  mat_float *src = nullptr;
  mat_float *dst = nullptr;
  std::vector<index_t> hrows((size_t)n + 1);
  std::vector<uint8_t> hmask((size_t)n, 1);
  uint8_t hperm[2] = {1, 0};
  std::vector<mat_float> hsrc((size_t)n * 4);
  for (int i = 0; i < n; ++i)
  {
    hrows[(size_t)i] = i;
    hsrc[(size_t)i * 4 + 0] = 1;
    hsrc[(size_t)i * 4 + 1] = 2;
    hsrc[(size_t)i * 4 + 2] = 3;
    hsrc[(size_t)i * 4 + 3] = 4;
  }
  hrows[(size_t)n] = n;
  if (hipMalloc(&rows, hrows.size() * sizeof(index_t)) != hipSuccess ||
      hipMalloc(&mask, hmask.size() * sizeof(uint8_t)) != hipSuccess ||
      hipMalloc(&perm, sizeof(hperm)) != hipSuccess ||
      hipMalloc(&src, hsrc.size() * sizeof(mat_float)) != hipSuccess ||
      hipMalloc(&dst, hsrc.size() * sizeof(mat_float)) != hipSuccess)
    return 2;
  if (hipMemcpy(rows, hrows.data(), hrows.size() * sizeof(index_t), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(mask, hmask.data(), hmask.size(), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(perm, hperm, sizeof(hperm), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(src, hsrc.data(), hsrc.size() * sizeof(mat_float), hipMemcpyHostToDevice) != hipSuccess)
    return 2;
  const int block = 256;
  const int grid = (n + block - 1) / block;
  auto launch = [&]() {
    brp_values_kernel<2><<<grid, block>>>(n, mask, rows, perm, src, dst);
  };
  const int rc = od_port_time(warmup, samples, launch, median_ms);
  if (rc == 0 &&
      hipMemcpy(hsrc.data(), dst, hsrc.size() * sizeof(mat_float), hipMemcpyDeviceToHost) != hipSuccess)
    return 2;
  // Weight by position so a copied block is not confused with a permuted one.
  double sum = 0;
  if (rc == 0)
    for (size_t k = 0; k < hsrc.size(); ++k)
      sum += (double)hsrc[k] * (double)((k % 4) + 1);
  if (checksum)
    *checksum = sum;
  hipFree(rows);
  hipFree(mask);
  hipFree(perm);
  hipFree(src);
  hipFree(dst);
  return rc;
}
#endif
  } // namespace linear_solvers
} // namespace opendarts
