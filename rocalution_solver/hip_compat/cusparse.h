#ifndef OD_HIP_CUSPARSE_H
#define OD_HIP_CUSPARSE_H

// Types named by the GPU solver headers. The HIP device library compiles
// those headers so the kernels can see the matrix layout. It does not call
// cuSPARSE.

using cusparseHandle_t = void *;
using cusparseMatDescr_t = void *;
using cusparseStatus_t = int;
using cusparseSolvePolicy_t = int;
using cusparseOperation_t = int;
using cusparseDirection_t = int;

constexpr int CUSPARSE_SOLVE_POLICY_USE_LEVEL = 0;
constexpr int CUSPARSE_OPERATION_NON_TRANSPOSE = 0;
constexpr int CUSPARSE_DIRECTION_ROW = 0;

#endif
