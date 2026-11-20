#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

// Thread-per-operator multilinear interpolation, one operator of one state per thread.
// N_DIMS is pressure plus three compositions. Each hypercube has 16 vertices and 56 operators.
// Stage 1 runs the states whose hypercube key is the sentinel.

constexpr int OBL_N_DIMS = 4;
constexpr int OBL_N_OPS = 56;
constexpr int OBL_N_VERTS = 1 << OBL_N_DIMS;
constexpr int OBL_VECTOR = OBL_N_VERTS * OBL_N_OPS;

struct OblKey
{
  int32_t idx[OBL_N_DIMS];
};

struct OblSlot
{
  int64_t key;
  double vector[OBL_VECTOR];
};

struct OblMap
{
  int size;
  int sizeMinus1;
  int occupied;
  OblSlot *data;
};

__device__ __forceinline__ uint64_t obl_mix(uint64_t h)
{
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}

__device__ __forceinline__ int64_t obl_key_from_cell(const OblKey &k)
{
  uint64_t h = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < OBL_N_DIMS; ++i)
  {
    const uint64_t v = static_cast<uint64_t>(static_cast<uint32_t>(k.idx[i]));
    h ^= obl_mix(v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
  }
  h = obl_mix(h);
  return static_cast<int64_t>(h & 0x7fffffffffffffffULL);
}

__device__ __forceinline__ int32_t obl_axis_index(double axis_value, double axis_origin, double axis_step_inv)
{
  const double scaled = (axis_value - axis_origin) * axis_step_inv;
  const double floored = floor(scaled);
  const double i32_min = static_cast<double>(std::numeric_limits<int32_t>::min());
  const double i32_max = static_cast<double>(std::numeric_limits<int32_t>::max());
  const double clamped = fmin(fmax(floored, i32_min), i32_max);
  return static_cast<int32_t>(clamped);
}

__device__ __forceinline__ int obl_lookup(OblMap *hashmap, int64_t key, double **vector)
{
  int slot = static_cast<int>(static_cast<uint64_t>(key) & static_cast<uint64_t>(hashmap->sizeMinus1));
  const int64_t empty_key = static_cast<int64_t>(-1);
  for (int i = 0; i < hashmap->size; i++)
  {
    if (hashmap->data[slot].key == key)
    {
      *vector = hashmap->data[slot].vector;
      return 0;
    }
    if (hashmap->data[slot].key == empty_key)
      return -1;
    slot = (slot + 1) & hashmap->sizeMinus1;
  }
  return -1;
}

__device__ __forceinline__ void obl_interpolate_operator(const double *axis_values, const double *body_data,
                                                         const double *axis_low, const double *axis_mult,
                                                         const double *axis_step_inv, int operator_idx,
                                                         double *interp_values, double *interp_derivs)
{
  uint32_t pwr = OBL_N_VERTS / 2;
  double workspace[2 * OBL_N_VERTS - 1];

  for (int i = 0; i < OBL_N_VERTS; ++i)
    workspace[i] = body_data[OBL_N_OPS * i + operator_idx];

  for (int i = 0; i < OBL_N_DIMS; ++i)
  {
    for (int j = 0; j < static_cast<int>(pwr); ++j)
    {
      workspace[2 * OBL_N_VERTS - (OBL_N_VERTS >> i) + j] =
          (workspace[j + static_cast<int>(pwr)] - workspace[j]) * axis_step_inv[i];

      for (int k = 0; k < i; k++)
      {
        workspace[2 * OBL_N_VERTS - (OBL_N_VERTS >> k) + j] =
            workspace[2 * OBL_N_VERTS - (OBL_N_VERTS >> k) + j] +
            axis_mult[i] * (workspace[2 * OBL_N_VERTS - (OBL_N_VERTS >> k) + j + static_cast<int>(pwr)] -
                            workspace[2 * OBL_N_VERTS - (OBL_N_VERTS >> k) + j]);
      }

      workspace[j] = workspace[j] + (axis_values[i] - axis_low[i]) * workspace[2 * OBL_N_VERTS - (OBL_N_VERTS >> i) + j];
    }
    pwr /= 2;
  }
  interp_values[operator_idx] = workspace[0];
  for (int i = 0; i < OBL_N_DIMS; ++i)
    interp_derivs[operator_idx * OBL_N_DIMS + i] = workspace[2 * OBL_N_VERTS - (OBL_N_VERTS >> i)];
}

__global__ void obl_interpolate(unsigned n_states_idxs, const int *states_idxs, const double *states,
                                const double *axis_min, const double *axis_step, const double *axis_step_inv,
                                OblMap *hypercubes, const OblKey *state_hc_keys, double *values, double *derivatives)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned operator_idx = i % OBL_N_OPS;
  const unsigned state_idx_idx = i / OBL_N_OPS;
  if (state_idx_idx > n_states_idxs - 1)
    return;

  if (state_hc_keys[state_idx_idx].idx[0] != INT32_MIN)
    return;

  const int state_idx = states_idxs[state_idx_idx];

  double axis_low[OBL_N_DIMS];
  double mult[OBL_N_DIMS];
  OblKey hc_key;
  for (int d = 0; d < OBL_N_DIMS; ++d)
  {
    hc_key.idx[d] = obl_axis_index(states[state_idx * OBL_N_DIMS + d], axis_min[d], axis_step_inv[d]);
    axis_low[d] = static_cast<double>(hc_key.idx[d]) * axis_step[d] + axis_min[d];
    mult[d] = (states[state_idx * OBL_N_DIMS + d] - axis_low[d]) * axis_step_inv[d];
  }

  double *hypercube_data = nullptr;
  if (obl_lookup(hypercubes, obl_key_from_cell(hc_key), &hypercube_data))
  {
    printf("Thread %u error s1: hypercube missing\n", i);
    return;
  }

  obl_interpolate_operator(states + state_idx * OBL_N_DIMS, hypercube_data, axis_low, mult, axis_step_inv, operator_idx,
                           values + state_idx * OBL_N_OPS, derivatives + state_idx * OBL_N_OPS * OBL_N_DIMS);
}
