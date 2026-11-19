#ifndef OD_HIP_CUDA_RUNTIME_H
#define OD_HIP_CUDA_RUNTIME_H

// CUDA runtime names used by the device kernels, implemented with HIP.
// This header is on the include path only for the HIP device library, ahead
// of any real CUDA headers. A CUDA build never sees it.

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;

#define cudaSuccess hipSuccess
#define cudaErrorHostMemoryAlreadyRegistered hipErrorHostMemoryAlreadyRegistered
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorNotSupported hipErrorNotSupported

#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaHostRegisterDefault hipHostRegisterDefault

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaMemGetInfo hipMemGetInfo
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

#endif
