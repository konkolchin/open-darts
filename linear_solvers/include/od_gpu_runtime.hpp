#ifndef OD_GPU_RUNTIME_HPP
#define OD_GPU_RUNTIME_HPP

// AMD GPU builds call the HIP runtime. hip_compat's cuda_runtime.h, which is
// first on the include path only for that configuration, turns the historical
// runtime names in shared device sources into hipMalloc, hipMemcpy and hipMemset.
// An NVIDIA build includes the CUDA runtime.

#ifdef OD_HIP_GPU
#include <hip/hip_runtime.h>
#endif
#include <cuda_runtime.h>

#endif
