#ifndef OD_HIP_CUDA_RUNTIME_H
#define OD_HIP_CUDA_RUNTIME_H

// CUDA runtime names used by the device kernels, implemented with HIP.
// This header is on the include path only for the HIP device library, ahead
// of any real CUDA headers. A CUDA build never sees it.

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;

constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;

inline cudaError_t cudaMalloc(void **ptr, size_t bytes) { return hipMalloc(ptr, bytes); }
inline cudaError_t cudaFree(void *ptr) { return hipFree(ptr); }
inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, hipMemcpyKind kind)
{
  return hipMemcpy(dst, src, bytes, kind);
}
inline cudaError_t cudaMemset(void *ptr, int value, size_t bytes) { return hipMemset(ptr, value, bytes); }
inline cudaError_t cudaDeviceSynchronize() { return hipDeviceSynchronize(); }
inline cudaError_t cudaGetLastError() { return hipGetLastError(); }
inline const char *cudaGetErrorString(cudaError_t err) { return hipGetErrorString(err); }
inline cudaError_t cudaHostUnregister(void *ptr) { return hipHostUnregister(ptr); }
inline cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
  return hipMemGetInfo(free_bytes, total_bytes);
}

#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

#endif
