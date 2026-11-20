// Standalone replay of one multilinear interpolation launch.
// The kernel and its device helpers are in obl_kernel.hip.
//
// gfx1100 check:
//   hipcc -O3 --offload-arch=gfx1100 bench_obl.hip obl_kernel.hip -o bench_obl
// Instinct MI300 (CDNA3) build line. A gfx1100 median is not an MI300X speedup:
//   hipcc -O3 --offload-arch=gfx942 bench_obl.hip obl_kernel.hip -o bench_obl
//
// rocprofv3 on gfx1100, enlarged 3ph_comp_w, one mid-run launch of
// multilinear_adaptive_interpolate_thread_per_operator_stages_kernel<double,4,56,true>:
//   Grid_Size 56000000 x 1 x 1, Workgroup_Size 128 x 1 x 1, LDS_Block_Size 0, Stream_Id 0.
// That kernel's total device time was 19732 us against 19038 us for assemble_jacobian_array_kernel.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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

__global__ void obl_interpolate(unsigned n_states_idxs, const int *states_idxs, const double *states,
                                const double *axis_min, const double *axis_step, const double *axis_step_inv,
                                OblMap *hypercubes, const OblKey *state_hc_keys, double *values, double *derivatives);

struct Scalars
{
  uint32_t magic;
  uint32_t n_states_idxs;
  uint32_t n_state_slots;
  uint32_t map_slots;
  uint32_t map_occupied;
  uint32_t n_dims;
  uint32_t n_ops;
  uint32_t block_size;
  uint32_t first_stage;
  uint32_t slot_bytes;
  uint32_t key_bytes;
  uint32_t map_bytes;
  double axes_origin[4];
  double axes_step[4];
  double axes_step_inv[4];
};

#define HIP_CHECK(call)                                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    hipError_t _err = (call);                                                                                          \
    if (_err != hipSuccess)                                                                                            \
    {                                                                                                                  \
      std::fprintf(stderr, "HIP %s\n", hipGetErrorString(_err));                                                      \
      return 1;                                                                                                        \
    }                                                                                                                  \
  } while (0)

static std::vector<char> read_file(const std::string &path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    std::fprintf(stderr, "missing %s\n", path.c_str());
    std::exit(1);
  }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> buf(static_cast<size_t>(n));
  if (n > 0)
    in.read(buf.data(), n);
  return buf;
}

int main(int argc, char **argv)
{
  const std::string dir = (argc > 1) ? argv[1] : ".";
  const std::vector<char> scalar_bytes = read_file(dir + "/scalars.bin");
  if (scalar_bytes.size() != sizeof(Scalars))
  {
    std::fprintf(stderr, "scalars.bin size %zu\n", scalar_bytes.size());
    return 1;
  }
  Scalars scalars{};
  std::memcpy(&scalars, scalar_bytes.data(), sizeof(scalars));
  if (scalars.magic != 0x4F424C31u || scalars.n_dims != 4 || scalars.n_ops != 56 || scalars.block_size != 128 ||
      scalars.slot_bytes != sizeof(OblSlot) || scalars.key_bytes != sizeof(OblKey) ||
      scalars.map_bytes != sizeof(OblMap))
  {
    std::fprintf(stderr, "snapshot layout mismatch slot=%u key=%u map=%u\n", scalars.slot_bytes, scalars.key_bytes,
                 scalars.map_bytes);
    return 1;
  }

  const std::vector<char> idx_bytes = read_file(dir + "/states_idxs.bin");
  const std::vector<char> state_bytes = read_file(dir + "/states.bin");
  const std::vector<char> key_bytes = read_file(dir + "/hc_keys.bin");
  const std::vector<char> slot_bytes = read_file(dir + "/hashmap_slots.bin");
  const std::vector<char> value_bytes = read_file(dir + "/values.bin");
  const std::vector<char> deriv_bytes = read_file(dir + "/derivatives.bin");

  const size_t n_states = scalars.n_states_idxs;
  const size_t n_slots = scalars.n_state_slots;
  if (idx_bytes.size() != n_states * sizeof(int) || state_bytes.size() != n_slots * 4 * sizeof(double) ||
      key_bytes.size() != n_states * sizeof(OblKey) ||
      slot_bytes.size() != static_cast<size_t>(scalars.map_slots) * sizeof(OblSlot) ||
      value_bytes.size() != n_slots * 56 * sizeof(double) ||
      deriv_bytes.size() != n_slots * 56 * 4 * sizeof(double))
  {
    std::fprintf(stderr, "snapshot array size mismatch\n");
    return 1;
  }

  int *d_idxs = nullptr;
  double *d_states = nullptr;
  double *d_axes = nullptr;
  OblSlot *d_slots = nullptr;
  OblMap *d_map = nullptr;
  OblKey *d_keys = nullptr;
  double *d_values = nullptr;
  double *d_derivs = nullptr;

  HIP_CHECK(hipMalloc(&d_idxs, idx_bytes.size()));
  HIP_CHECK(hipMalloc(&d_states, state_bytes.size()));
  HIP_CHECK(hipMalloc(&d_axes, 12 * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_slots, slot_bytes.size()));
  HIP_CHECK(hipMalloc(&d_map, sizeof(OblMap)));
  HIP_CHECK(hipMalloc(&d_keys, key_bytes.size()));
  HIP_CHECK(hipMalloc(&d_values, value_bytes.size()));
  HIP_CHECK(hipMalloc(&d_derivs, deriv_bytes.size()));

  HIP_CHECK(hipMemcpy(d_idxs, idx_bytes.data(), idx_bytes.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_states, state_bytes.data(), state_bytes.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_slots, slot_bytes.data(), slot_bytes.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_keys, key_bytes.data(), key_bytes.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_values, 0, value_bytes.size()));
  HIP_CHECK(hipMemset(d_derivs, 0, deriv_bytes.size()));

  double axes[12];
  for (int i = 0; i < 4; ++i)
  {
    axes[i] = scalars.axes_origin[i];
    axes[4 + i] = scalars.axes_step[i];
    axes[8 + i] = scalars.axes_step_inv[i];
  }
  HIP_CHECK(hipMemcpy(d_axes, axes, sizeof(axes), hipMemcpyHostToDevice));

  OblMap host_map{};
  host_map.size = static_cast<int>(scalars.map_slots);
  host_map.sizeMinus1 = host_map.size - 1;
  host_map.occupied = static_cast<int>(scalars.map_occupied);
  host_map.data = d_slots;
  HIP_CHECK(hipMemcpy(d_map, &host_map, sizeof(host_map), hipMemcpyHostToDevice));

  const unsigned n_threads = static_cast<unsigned>(n_states) * 56u;
  const unsigned block = scalars.block_size;
  const unsigned grid = (n_threads + block - 1) / block;

  auto launch = [&]() {
    obl_interpolate<<<grid, block>>>(static_cast<unsigned>(n_states), d_idxs, d_states, d_axes, d_axes + 4, d_axes + 8,
                                     d_map, d_keys, d_values, d_derivs);
  };

  for (int i = 0; i < 10; ++i)
    launch();
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> samples(100);
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  for (int i = 0; i < 100; ++i)
  {
    HIP_CHECK(hipEventRecord(start));
    launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&samples[static_cast<size_t>(i)], start, stop));
  }
  std::sort(samples.begin(), samples.end());
  const double median_ms = 0.5 * (static_cast<double>(samples[49]) + static_cast<double>(samples[50]));
  const double median_us = median_ms * 1000.0;

  std::vector<double> got_values(n_slots * 56);
  std::vector<double> got_derivs(n_slots * 56 * 4);
  HIP_CHECK(hipMemcpy(got_values.data(), d_values, value_bytes.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_derivs.data(), d_derivs, deriv_bytes.size(), hipMemcpyDeviceToHost));

  const double *gold_values = reinterpret_cast<const double *>(value_bytes.data());
  const double *gold_derivs = reinterpret_cast<const double *>(deriv_bytes.data());
  const int *idxs = reinterpret_cast<const int *>(idx_bytes.data());
  const int32_t *keys = reinterpret_cast<const int32_t *>(key_bytes.data());

  double max_err = 0.0;
  uint64_t n_compared = 0;
  int bad = 0;
  for (size_t s = 0; s < n_states; ++s)
  {
    if (keys[s * 4] != INT32_MIN)
      continue;
    const int state = idxs[s];
    if (state < 0 || static_cast<size_t>(state) >= n_slots)
    {
      bad = 1;
      break;
    }
    for (int op = 0; op < 56; ++op)
    {
      const size_t vpos = static_cast<size_t>(state) * 56 + static_cast<size_t>(op);
      const double g = gold_values[vpos];
      const double y = got_values[vpos];
      if (!std::isfinite(g) || !std::isfinite(y))
      {
        bad = 1;
        break;
      }
      const double abs_g = std::fabs(g);
      const double err = (abs_g < 1.0) ? std::fabs(y - g) : std::fabs(y - g) / abs_g;
      if (err > max_err)
        max_err = err;
      ++n_compared;
      for (int d = 0; d < 4; ++d)
      {
        const size_t dpos = vpos * 4 + static_cast<size_t>(d);
        const double gd = gold_derivs[dpos];
        const double yd = got_derivs[dpos];
        if (!std::isfinite(gd) || !std::isfinite(yd))
        {
          bad = 1;
          break;
        }
        const double abs_gd = std::fabs(gd);
        const double errd = (abs_gd < 1.0) ? std::fabs(yd - gd) : std::fabs(yd - gd) / abs_gd;
        if (errd > max_err)
          max_err = errd;
        ++n_compared;
      }
      if (bad)
        break;
    }
    if (bad)
      break;
  }
  if (n_compared == 0 || max_err > 1e-12)
    bad = 1;

  const double payload =
      static_cast<double>(idx_bytes.size() + state_bytes.size() + key_bytes.size() + slot_bytes.size() +
                          value_bytes.size() + deriv_bytes.size() + sizeof(axes));
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, 0);
  std::printf("DEVICE %s\n", prop.name);
  std::printf("ARCH %s\n", prop.gcnArchName);
  std::printf("GLOBAL %u 1 1\n", n_threads);
  std::printf("BLOCK %u 1 1\n", block);
  std::printf("DYNAMIC_SHARED 0\n");
  std::printf("STREAM 0\n");
  std::printf("WARMUP 10\n");
  std::printf("SAMPLES 100\n");
  std::printf("MEDIAN_US %.3f\n", median_us);
  std::printf("PAYLOAD_BYTES %.0f\n", payload);
  std::printf("BYTES_PER_S %.6e\n", payload / (median_us * 1e-6));
  std::printf("MAX_ERR %.6e\n", max_err);
  std::printf("N_COMPARED %llu\n", static_cast<unsigned long long>(n_compared));
  if (bad)
  {
    std::printf("CHECK_FAIL\n");
    return 1;
  }
  std::printf("CHECK_OK\n");
  return 0;
}
