#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "multilinear_adaptive_gpu_interpolator.hpp"
#include "gpu_tools.h"

#define USE_THREAD_PER_OPERATOR_KERNEL
#define HYPERCUBE_BUFFER_SIZE 100
// Capacity requested for a new device hypercube map; create_hashmap() rounds it up to 2048 slots. The map doubles
// before an insert would take it above a 0.7 load factor, so it can start small. add_hypercubes_to_hashmap()
// threads may read the occupancy after their batch was already added to it, so the slot count must stay above
// HYPERCUBE_BUFFER_SIZE / 0.3.
#define HYPERCUBE_MAP_INITIAL_CAPACITY 1024

// ─── kernel forward declarations ───────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void multilinear_adaptive3_check_hypercube_ready_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    cell_key_t<N_DIMS> *state_hc_keys, int *batch_status_d);

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_state_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d);

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_operator_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d);

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void add_hypercubes_to_hashmap(
    const unsigned int n_new_hypercubes, const cell_key_t<N_DIMS> *new_hypercube_keys_d,
    const value_t *new_hypercube_data_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d);

// ─── constructor / destructor / init ───────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::multilinear_adaptive_gpu_interpolator(
    operator_set_evaluator_iface *supporting_point_evaluator,
    const std::vector<double> &axes_origin,
    const std::vector<double> &axes_step)
    : multilinear_gpu_interpolator_base<uint64_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_origin, axes_step)
{
  // A multiple of the warp size: the check kernel counts missing hypercubes with a full-warp ballot.
  this->kernel_block_size = 128;

  create_hypercube_map();

  cudaStreamCreate(&hypercube_generation_stream);
  cudaEventCreateWithFlags(&batch_status_ready, cudaEventDisableTiming);

  new_hypercube_data_buffer.resize(HYPERCUBE_BUFFER_SIZE * N_OPS * N_VERTS);
  new_hypercube_index_buffer.resize(HYPERCUBE_BUFFER_SIZE);
  new_hypercube_data_buffer_d.resize(HYPERCUBE_BUFFER_SIZE * N_OPS * N_VERTS);
  new_hypercube_index_buffer_d.resize(HYPERCUBE_BUFFER_SIZE);
  batch_status_host.resize(2);
  batch_status_d.resize(2);
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::~multilinear_adaptive_gpu_interpolator()
{
  gpu_hashmap_async::delete_hashmap(hypercube_data_d);
  cudaEventDestroy(batch_status_ready);
  cudaStreamDestroy(hypercube_generation_stream);
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::create_hypercube_map()
{
  hypercube_data_d = gpu_hashmap_async::create_hashmap<value_t, N_VERTS * N_OPS>(HYPERCUBE_MAP_INITIAL_CAPACITY);
  gpu_hashmap_async::gpu_hash_map<value_t, N_VERTS * N_OPS> header;
  cudaMemcpy(&header, hypercube_data_d, sizeof(header), cudaMemcpyDeviceToHost);
  hashmap_size = header.size;
  hashmap_occupied = header.occupied;
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::init()
{
  return 0;
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::clear_hypercube_data()
{
  // Drop the device hypercube map and the host key tracker; both are rebuilt on
  // demand from point_data (no flash). The fresh map starts at the initial capacity,
  // as in the constructor. point_data and the on-disk cache are untouched.
  gpu_hashmap_async::delete_hashmap(hypercube_data_d);
  create_hypercube_map();
  generated_hypercubes.clear();
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::write_to_file(const std::string filename)
{
  return 0;
}

// ─── supporting-point access (host) ────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
const typename multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::point_data_t &
multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::get_point_data(const key_t &point_key)
{
  auto item = point_data.find(point_key);
  if (item != point_data.end())
    return item->second;

  // Compute physical coordinates from multi-index, evaluate, store, return.
  point_data_t new_point;
  for (uint8_t i = 0; i < N_DIMS; ++i)
    this->new_point_coords[i] = this->axes_origin[i] + this->axes_step[i] * static_cast<double>(point_key.idx[i]);
  this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
  for (int op = 0; op < N_OPS; op++)
  {
    new_point[op] = this->new_operator_values[op];
    // isfinite (not just isnan): an inf operator poisons interpolation like a nan
    if (!std::isfinite(this->new_operator_values[op]))
    {
      printf("OBL generation warning: non-finite operator detected! Operator %d for point (", op);
      for (int a = 0; a < N_DIMS; a++)
        printf("%lf, ", this->new_point_coords[a]);
      printf(") is %lf\n", this->new_operator_values[op]);
    }
  }
  auto insert_result = point_data.emplace(point_key, new_point);
  // Mark for append-only cache flush after this new point is materialized.
  dirty_point_data.insert(point_key);
  // Stamp the point with the current batch-interpolation (nonlinear-iteration) index.
  dirty_point_epochs[point_key] = eval_index;
  this->n_points_used++;
  return insert_result.first->second;
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::generate_hypercube(
    const key_t &hc_key, value_t *new_hypercube)
{
  // Build the N_VERTS supporting-point keys from the lower corner.
  static const uint32_t NV = (1u << N_DIMS);
  for (uint32_t v = 0; v < NV; ++v)
  {
    key_t pt_key;
    for (uint8_t i = 0; i < N_DIMS; ++i)
    {
      const int32_t bit = static_cast<int32_t>((v >> (N_DIMS - 1 - i)) & 1u);
      pt_key.idx[i] = hc_key.idx[i] + bit;
    }
    const point_data_t &p_data = this->get_point_data(pt_key);
    for (int op = 0; op < N_OPS; op++)
      new_hypercube[v * N_OPS + op] = p_data[op];
  }
  return 0;
}

// ─── one-shot Hyperloom snapshot ───────────────────────────────────────────────

// Copies one interpolation launch to raw bins when OD_HYPERLOOM_DUMP=1. Unset, this is a no-op.
template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void od_hyperloom_dump_interpolation_launch(
    bool first_stage, int n_states_idxs, int block_size,
    double *states_d, int *states_idxs_d, double *values_d, double *derivatives_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys)
{
  const char *flag = std::getenv("OD_HYPERLOOM_DUMP");
  if (flag == nullptr || std::strcmp(flag, "1") != 0)
    return;
  if (!(first_stage && N_DIMS == 4 && N_OPS == 56 && n_states_idxs >= 100000))
    return;

  static bool dumped = false;
  if (dumped)
    return;

  cudaDeviceSynchronize();

  const char *dir_env = std::getenv("OD_HYPERLOOM_DIR");
  const std::string dir = (dir_env != nullptr && dir_env[0] != '\0')
                               ? std::string(dir_env)
                               : std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/od-obl-snapshot";
  std::filesystem::create_directories(dir);

  std::vector<cell_key_t<N_DIMS>> keys(static_cast<size_t>(n_states_idxs));
  cudaMemcpy(keys.data(), state_hc_keys, keys.size() * sizeof(cell_key_t<N_DIMS>), cudaMemcpyDeviceToHost);
  int n_sentinel = 0;
  for (const cell_key_t<N_DIMS> &key : keys)
  {
    if (key.idx[0] == INT32_MIN)
      ++n_sentinel;
  }
  // The first large stage-1 launch is the cache miss and returns without writing operators.
  // Keep that launch out of the snapshot; the next one interpolates the cached hypercubes.
  if (n_sentinel * 2 < n_states_idxs)
    return;
  dumped = true;

  std::vector<int> idxs(static_cast<size_t>(n_states_idxs));
  cudaMemcpy(idxs.data(), states_idxs_d, idxs.size() * sizeof(int), cudaMemcpyDeviceToHost);
  int max_idx = 0;
  for (int idx : idxs)
  {
    if (idx > max_idx)
      max_idx = idx;
  }
  const uint32_t n_state_slots = static_cast<uint32_t>(max_idx) + 1;

  using map_t = gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS>;
  using slot_t = gpu_hashmap_async::key_vector<value_t, (1 << N_DIMS) * N_OPS>;
  map_t host_map{};
  cudaMemcpy(&host_map, hypercube_data_d, sizeof(host_map), cudaMemcpyDeviceToHost);

  auto write_device = [&](const std::string &name, const void *src, size_t bytes) {
    std::ofstream out(dir + "/" + name, std::ios::binary);
    const size_t chunk = 64u * 1024u * 1024u;
    std::vector<char> buf(std::min(chunk, std::max<size_t>(bytes, 1)));
    for (size_t off = 0; off < bytes; off += chunk)
    {
      const size_t n = std::min(chunk, bytes - off);
      cudaMemcpy(buf.data(), static_cast<const char *>(src) + off, n, cudaMemcpyDeviceToHost);
      out.write(buf.data(), static_cast<std::streamsize>(n));
    }
  };

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
  Scalars scalars{};
  scalars.magic = 0x4F424C31u;
  scalars.n_states_idxs = static_cast<uint32_t>(n_states_idxs);
  scalars.n_state_slots = n_state_slots;
  scalars.map_slots = static_cast<uint32_t>(host_map.size);
  scalars.map_occupied = static_cast<uint32_t>(host_map.occupied);
  scalars.n_dims = N_DIMS;
  scalars.n_ops = N_OPS;
  scalars.block_size = static_cast<uint32_t>(block_size);
  scalars.first_stage = 1;
  scalars.slot_bytes = static_cast<uint32_t>(sizeof(slot_t));
  scalars.key_bytes = static_cast<uint32_t>(sizeof(cell_key_t<N_DIMS>));
  scalars.map_bytes = static_cast<uint32_t>(sizeof(map_t));
  cudaMemcpy(scalars.axes_origin, axis_min_d, sizeof(scalars.axes_origin), cudaMemcpyDeviceToHost);
  cudaMemcpy(scalars.axes_step, axis_step_d, sizeof(scalars.axes_step), cudaMemcpyDeviceToHost);
  cudaMemcpy(scalars.axes_step_inv, axis_step_inv_d, sizeof(scalars.axes_step_inv), cudaMemcpyDeviceToHost);

  {
    std::ofstream out(dir + "/scalars.bin", std::ios::binary);
    out.write(reinterpret_cast<const char *>(&scalars), sizeof(scalars));
  }
  {
    std::ofstream out(dir + "/states_idxs.bin", std::ios::binary);
    out.write(reinterpret_cast<const char *>(idxs.data()), static_cast<std::streamsize>(idxs.size() * sizeof(int)));
  }
  write_device("states.bin", states_d, static_cast<size_t>(n_state_slots) * N_DIMS * sizeof(double));
  write_device("hc_keys.bin", state_hc_keys, static_cast<size_t>(n_states_idxs) * sizeof(cell_key_t<N_DIMS>));
  write_device("hashmap_slots.bin", host_map.data, static_cast<size_t>(host_map.size) * sizeof(slot_t));
  write_device("values.bin", values_d, static_cast<size_t>(n_state_slots) * N_OPS * sizeof(double));
  write_device("derivatives.bin", derivatives_d, static_cast<size_t>(n_state_slots) * N_OPS * N_DIMS * sizeof(double));

  const unsigned n_threads = static_cast<unsigned>(n_states_idxs) * N_OPS;
  const unsigned grid_x = (n_threads + static_cast<unsigned>(block_size) - 1) / static_cast<unsigned>(block_size);
  std::ofstream launch(dir + "/launch.txt");
  launch << "kernel multilinear_adaptive_interpolate_thread_per_operator_stages_kernel<double,4,56,true>\n"
         << "global " << n_threads << " 1 1\n"
         << "grid " << grid_x << " 1 1\n"
         << "block " << block_size << " 1 1\n"
         << "dynamic_shared 0\n"
         << "stream 0\n";
  launch.flush();
  std::cout << "OD_HYPERLOOM_DUMP_OK dir=" << dir << " states=" << n_states_idxs
            << " global=" << n_threads << " block=" << block_size << " lds=0 stream=0" << std::endl;
}

// ─── batch evaluation ──────────────────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
template <bool FIRST_STAGE>
void multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::launch_interpolation_stage(
    int n_states_idxs, double *states_d, int *states_idxs_d, double *values_d, double *derivatives_d)
{
#ifdef USE_THREAD_PER_OPERATOR_KERNEL
  multilinear_adaptive_interpolate_thread_per_operator_stages_kernel<value_t, N_DIMS, N_OPS, FIRST_STAGE>
      KERNEL_1D_THREAD(n_states_idxs * N_OPS, this->kernel_block_size)(
          n_states_idxs, states_idxs_d, states_d,
          thrust::raw_pointer_cast(this->axes_origin_d.data()),
          thrust::raw_pointer_cast(this->axes_step_d.data()),
          thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
          hypercube_data_d,
          thrust::raw_pointer_cast(state_hc_keys_d.data()),
          values_d, derivatives_d);
#else
  multilinear_adaptive_interpolate_thread_per_state_stages_kernel<value_t, N_DIMS, N_OPS, FIRST_STAGE>
      KERNEL_1D_THREAD(n_states_idxs, this->kernel_block_size)(
          n_states_idxs, states_idxs_d, states_d,
          thrust::raw_pointer_cast(this->axes_origin_d.data()),
          thrust::raw_pointer_cast(this->axes_step_d.data()),
          thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
          hypercube_data_d,
          thrust::raw_pointer_cast(state_hc_keys_d.data()),
          values_d, derivatives_d);
#endif
  od_hyperloom_dump_interpolation_launch<value_t, N_DIMS, N_OPS>(
      FIRST_STAGE, n_states_idxs, this->kernel_block_size, states_d, states_idxs_d, values_d, derivatives_d,
      thrust::raw_pointer_cast(this->axes_origin_d.data()),
      thrust::raw_pointer_cast(this->axes_step_d.data()),
      thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
      hypercube_data_d, thrust::raw_pointer_cast(state_hc_keys_d.data()));
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::evaluate_with_derivatives_d(
    int n_states_idxs, double *states_d, int *states_idxs_d,
    double *values_d, double *derivatives_d)
{
  this->timer->start();
  this->timer->node["gpu interpolation"].start();

  // One batch interpolation call == one nonlinear-iteration assembly of this operator
  // set: advance the epoch stamp applied to points materialized during this call.
  eval_index++;

  if (n_states_idxs <= 0)
  {
    this->timer->node["gpu interpolation"].stop();
    this->timer->stop();
    return 0;
  }

  // Bound the device hypercube cache: when it exceeds hypercube_cap, drop all cached
  // hypercubes (host tracker + device map) and let this batch rebuild only its working
  // set from point_data (no flash). Coarse vs the CPU LRU because the device
  // open-addressed map has no selective-erase API; the per-batch working set is
  // bounded by n_states_idxs, so size hypercube_cap >> cells.
  if (hypercube_cap != 0 && generated_hypercubes.size() > hypercube_cap)
    clear_hypercube_data();

  state_hc_keys_d.resize(n_states_idxs);

  // Look up every state's hypercube. The check kernel counts the states whose hypercube
  // is missing and the int32 cell-index overflows into batch_status_d; only these two
  // counters are copied to the host. The first-stage kernel (states whose hypercube is
  // cached) is queued behind that copy, so it runs while the host waits for the status.
  int *batch_status_d_ptr = thrust::raw_pointer_cast(batch_status_d.data());
  const int *batch_status = thrust::raw_pointer_cast(batch_status_host.data());
  cudaMemsetAsync(batch_status_d_ptr, 0, 2 * sizeof(int));
  multilinear_adaptive3_check_hypercube_ready_kernel<value_t, N_DIMS, N_OPS>
      KERNEL_1D_THREAD(n_states_idxs, this->kernel_block_size)(
          n_states_idxs, states_idxs_d, states_d,
          thrust::raw_pointer_cast(this->axes_origin_d.data()),
          thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
          hypercube_data_d,
          thrust::raw_pointer_cast(state_hc_keys_d.data()),
          batch_status_d_ptr);
  cudaMemcpyAsync(thrust::raw_pointer_cast(batch_status_host.data()), batch_status_d_ptr, 2 * sizeof(int),
                  cudaMemcpyDeviceToHost);
  cudaEventRecord(batch_status_ready);
  launch_interpolation_stage<true>(n_states_idxs, states_d, states_idxs_d, values_d, derivatives_d);
  cudaEventSynchronize(batch_status_ready);
  const int n_missing_states = batch_status[0];
  const int n_axis_overflows = batch_status[1];

  if (n_missing_states == 0)
  {
    // Every state's hypercube was cached: the first stage interpolates them all.
    this->timer->node["gpu interpolation"].stop();
    this->report_axis_index_overflows(static_cast<uint64_t>(n_axis_overflows),
                                      this->axes_origin, this->axes_step);
    this->n_interpolations += N_OPS * n_states_idxs;
    this->timer->stop();
    return 0;
  }

  // Copy hypercube multi-indices to host (DTOH); cached states carry the sentinel.
  hypercubes_to_compute.resize(n_states_idxs);
  cudaMemcpyAsync(thrust::raw_pointer_cast(hypercubes_to_compute.data()),
                  thrust::raw_pointer_cast(state_hc_keys_d.data()),
                  n_states_idxs * sizeof(key_t),
                  cudaMemcpyDeviceToHost, hypercube_generation_stream);
  cudaStreamSynchronize(hypercube_generation_stream);

  // ── Batch pre-pass ───────────────────────────────────────────────────────────
  // Collect every UNIQUE missing supporting point across all missing hypercubes and
  // evaluate them in a SINGLE evaluate_batch() call, then populate point_data. This is
  // what the CPU adaptive interpolators do; it lets a ParallelEvaluator split the batch
  // across worker processes (parallel_evaluation=True). Without it, generate_hypercube()
  // below evaluates each missing point one at a time (get_point_data -> per-point
  // evaluate), which cannot parallelize and dominates init time. After this pass every
  // needed point is cached, so the generation loop is pure cache hits. Serial-safe: the
  // default evaluate_batch just loops evaluate() in C++.
  {
    static const uint32_t NV_pp = (1u << N_DIMS);
    std::unordered_set<key_t, key_hash_t> queued_pts;
    std::unordered_set<key_t, key_hash_t> seen_hcs;
    std::vector<key_t> missing_pts;
    for (int i = 0; i < n_states_idxs; i++)
    {
      const key_t &hc_key = hypercubes_to_compute[i];
      if (hc_key.idx[0] == INT32_MIN)             continue;  // sentinel: already cached
      if (generated_hypercubes.count(hc_key))     continue;  // already generated earlier
      if (!seen_hcs.insert(hc_key).second)        continue;  // duplicate within this batch
      for (uint32_t v = 0; v < NV_pp; ++v)
      {
        key_t pt_key;
        for (uint8_t d = 0; d < N_DIMS; ++d)
        {
          const int32_t bit = static_cast<int32_t>((v >> (N_DIMS - 1 - d)) & 1u);
          pt_key.idx[d] = hc_key.idx[d] + bit;
        }
        if (point_data.find(pt_key) != point_data.end()) continue;  // already cached
        if (!queued_pts.insert(pt_key).second)           continue;  // already queued
        missing_pts.push_back(pt_key);
      }
    }
    if (!missing_pts.empty())
    {
      const size_t n_missing = missing_pts.size();
      std::vector<double> batch_coords(n_missing * N_DIMS);
      for (size_t k = 0; k < n_missing; ++k)
        for (uint8_t d = 0; d < N_DIMS; ++d)
          batch_coords[k * N_DIMS + d] =
              this->axes_origin[d] + this->axes_step[d] * static_cast<double>(missing_pts[k].idx[d]);
      std::vector<double> batch_values(n_missing * N_OPS);
      this->supporting_point_evaluator->evaluate_batch(
          batch_coords, static_cast<int>(n_missing), batch_values, N_OPS);
      for (size_t k = 0; k < n_missing; ++k)
      {
        point_data_t np;
        for (int op = 0; op < N_OPS; op++)
        {
          np[op] = batch_values[k * N_OPS + op];
          // isfinite (not just isnan): an inf operator poisons interpolation like a nan
          if (!std::isfinite(batch_values[k * N_OPS + op]))
            printf("OBL generation warning: non-finite operator %d in batch-generated point\n", op);
        }
        point_data.emplace(missing_pts[k], np);
        dirty_point_data.insert(missing_pts[k]);
        dirty_point_epochs[missing_pts[k]] = eval_index;
        this->n_points_used++;
      }
    }
  }

  // Generate the missing hypercubes straight into the pinned staging buffers and insert
  // them into the device map in batches of up to HYPERCUBE_BUFFER_SIZE.
  bool new_hypercubes_generated = false;
  value_t *data_staging = thrust::raw_pointer_cast(new_hypercube_data_buffer.data());
  key_t *index_staging = thrust::raw_pointer_cast(new_hypercube_index_buffer.data());
  for (int i = 0, h = 0; i < n_states_idxs; i++)
  {
    const key_t &hc_key = hypercubes_to_compute[i];
    // sentinel: idx[0] == INT32_MIN means "already cached, skip"
    const bool is_sentinel = (hc_key.idx[0] == INT32_MIN);
    if (!is_sentinel && !generated_hypercubes.count(hc_key))
    {
      generated_hypercubes.insert(hc_key);
      index_staging[h] = hc_key;
      generate_hypercube(hc_key, data_staging + h * N_OPS * N_VERTS);
      new_hypercubes_generated = true;
      h++;
    }

    if (h == HYPERCUBE_BUFFER_SIZE || (i == (n_states_idxs - 1) && h))
    {
      // Ensure the map has room for THIS batch before inserting it, keeping occupied + h
      // under a 0.7 load factor; otherwise add_hypercubes_to_hashmap hits its "hashmap
      // overflow" branch and silently drops the excess, which later surfaces as
      // "hypercube missing". Size and occupancy come from the host mirror of the header.
      bool grew = false;
      while (static_cast<long long>(hashmap_occupied) + h > static_cast<long long>(0.7 * hashmap_size))
      {
        hypercube_data_d = gpu_hashmap_async::expand_hashmap(hypercube_data_d, 2);
        hashmap_size *= 2;
        grew = true;
      }
      if (grew)
        cudaDeviceSynchronize();  // rehash (default stream) must finish before the batch insert

      cudaMemcpyAsync(thrust::raw_pointer_cast(new_hypercube_data_buffer_d.data()), data_staging,
                      h * sizeof(value_t) * N_OPS * N_VERTS,
                      cudaMemcpyHostToDevice, hypercube_generation_stream);
      cudaMemcpyAsync(thrust::raw_pointer_cast(new_hypercube_index_buffer_d.data()), index_staging,
                      h * sizeof(key_t),
                      cudaMemcpyHostToDevice, hypercube_generation_stream);

      add_hypercubes_to_hashmap<value_t, N_DIMS, N_OPS>
          KERNEL_1D_THREAD_STREAM(h * N_OPS * N_VERTS, this->kernel_block_size, hypercube_generation_stream)(
              h, thrust::raw_pointer_cast(new_hypercube_index_buffer_d.data()),
              thrust::raw_pointer_cast(new_hypercube_data_buffer_d.data()),
              hypercube_data_d);
      hashmap_occupied += h;

      // The next batch refills the staging buffers: wait for the uploads and the insert.
      CUDA_CHECK_RETURN(cudaStreamSynchronize(hypercube_generation_stream));
      h = 0;
    }
  }

  // Second stage: the states whose hypercube was generated in this call. The inserts
  // completed above (host-synchronized), and the kernel runs on the default stream.
  if (new_hypercubes_generated)
    launch_interpolation_stage<false>(n_states_idxs, states_d, states_idxs_d, values_d, derivatives_d);

  this->timer->node["gpu interpolation"].stop();

  // Hand this batch's int32 cell-index overflow tally to the unified reporter (same
  // warning path as the CPU backend).
  this->report_axis_index_overflows(static_cast<uint64_t>(n_axis_overflows),
                                    this->axes_origin, this->axes_step);

  this->n_interpolations += N_OPS * n_states_idxs;
  this->timer->stop();
  return 0;
}

// ─── kernel definitions ────────────────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void multilinear_adaptive3_check_hypercube_ready_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    cell_key_t<N_DIMS> *state_hc_keys, int *batch_status_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  bool missing = false;
  if (i < n_states_idxs)
  {
    int state_idx = states_idxs_d[i];

    // Per-thread int32 cell-index overflow tally, accumulated branchlessly into a
    // register. This is the canonical index computation for the batch (mirrors CPU
    // Phase 1); the interpolate kernels re-derive the index without counting so cells
    // are tallied once. Reduced to the device accumulator below only if nonzero — a
    // warp-coherent, almost-always-not-taken branch, so effectively free on the hot path.
    int cell_overflows = 0;
    cell_key_t<N_DIMS> hc_key;
    for (int d = 0; d < N_DIMS; ++d)
    {
      hc_key.idx[d] = get_axis_interval_index_unbounded<value_t>(
          states_d[state_idx * N_DIMS + d], axis_min_d[d], axis_step_inv_d[d], &cell_overflows);
    }
    if (cell_overflows)
      atomicAdd(&batch_status_d[1], cell_overflows);

    value_t *hypercube_data;
    if (lookup_data(hypercube_data_d, gpu_hashmap_async::key_from_cell<N_DIMS>(hc_key), &hypercube_data))
    {
      // hypercube not available; write multi-index for stage-2 generation
      state_hc_keys[i] = hc_key;
      missing = true;
    }
    else
    {
      // hypercube cached; mark sentinel
      state_hc_keys[i].idx[0] = INT32_MIN;
    }
  }

  // Count the states with a missing hypercube once per warp. Every lane reaches the
  // ballot (there is no early return above), so the block size must be a multiple of 32.
#ifdef __HIPCC__
  const unsigned long long missing_lanes = __ballot_sync(~0ull, missing);
  if (missing_lanes != 0ull && (threadIdx.x & 63u) == 0u)
    atomicAdd(&batch_status_d[0], __popcll(missing_lanes));
#else
  const unsigned missing_lanes = __ballot_sync(0xffffffffu, missing);
  if (missing_lanes != 0u && (threadIdx.x & 31u) == 0u)
    atomicAdd(&batch_status_d[0], __popc(missing_lanes));
#endif
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_state_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n_states_idxs - 1)
    return;

  const bool is_sentinel = (state_hc_keys[i].idx[0] == INT32_MIN);
  if (FIRST_STAGE)
  {
    // Stage 1: process states whose hypercube is already cached
    if (!is_sentinel)
      return;
  }
  else
  {
    // Stage 2: process states whose hypercube was just generated
    if (is_sentinel)
      return;
  }

  int state_idx = states_idxs_d[i];

  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];
  cell_key_t<N_DIMS> hc_key;
  for (int d = 0; d < N_DIMS; ++d)
  {
    hc_key.idx[d] = get_axis_interval_index_low_mult_unbounded<value_t>(
        states_d[state_idx * N_DIMS + d],
        axis_min_d[d], axis_step_d[d], axis_step_inv_d[d],
        &axis_low[d], &mult[d]);
  }

  value_t *hypercube_data;
  // hashmap is keyed by the 63-bit content hash of cell_key_t (key_from_cell);
  // this GPU path is collision-PRONE, not collision-safe — unlike the CPU
  // std::unordered_map<cell_key_t> path. See gpu_hashmap_async.h for the tradeoff.
  if (lookup_data(hypercube_data_d, gpu_hashmap_async::key_from_cell<N_DIMS>(hc_key), &hypercube_data))
  {
    if (FIRST_STAGE)
      printf("Thread %u error s1: hypercube missing\n", i);
    else
      printf("Thread %u error s2: hypercube missing\n", i);
  }
  else
  {
    interpolate_point_with_derivatives<value_t, N_DIMS, N_OPS>(
        states_d + state_idx * N_DIMS, hypercube_data,
        &axis_low[0], &mult[0], axis_step_inv_d,
        values_d + state_idx * N_OPS,
        derivatives_d + state_idx * N_OPS * N_DIMS);
  }
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_operator_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned operator_idx = i % N_OPS;
  const unsigned state_idx_idx = i / N_OPS;
  if (state_idx_idx > n_states_idxs - 1)
    return;

  const bool is_sentinel = (state_hc_keys[state_idx_idx].idx[0] == INT32_MIN);
  if (FIRST_STAGE)
  {
    if (!is_sentinel)
      return;
  }
  else
  {
    if (is_sentinel)
      return;
  }

  int state_idx = states_idxs_d[state_idx_idx];

  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];
  cell_key_t<N_DIMS> hc_key;
  for (int d = 0; d < N_DIMS; ++d)
  {
    hc_key.idx[d] = get_axis_interval_index_low_mult_unbounded<value_t>(
        states_d[state_idx * N_DIMS + d],
        axis_min_d[d], axis_step_d[d], axis_step_inv_d[d],
        &axis_low[d], &mult[d]);
  }

  value_t *hypercube_data;
  // hashmap is keyed by the 63-bit content hash of cell_key_t (key_from_cell);
  // this GPU path is collision-PRONE, not collision-safe — unlike the CPU
  // std::unordered_map<cell_key_t> path. See gpu_hashmap_async.h for the tradeoff.
  if (lookup_data(hypercube_data_d, gpu_hashmap_async::key_from_cell<N_DIMS>(hc_key), &hypercube_data))
  {
    if (FIRST_STAGE)
      printf("Thread %u error s1: hypercube missing\n", i);
    else
      printf("Thread %u error s2: hypercube missing\n", i);
  }
  else
  {
    interpolate_operator_with_derivatives<value_t, N_DIMS, N_OPS>(
        states_d + state_idx * N_DIMS, hypercube_data,
        &axis_low[0], &mult[0], axis_step_inv_d,
        operator_idx,
        values_d + state_idx * N_OPS,
        derivatives_d + state_idx * N_OPS * N_DIMS);
  }
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void add_hypercubes_to_hashmap(
    const unsigned int n_new_hypercubes, const cell_key_t<N_DIMS> *new_hypercube_keys_d,
    const value_t *new_hypercube_data_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  static const uint32_t NV = (1u << N_DIMS);
  const unsigned hypercube_index = i / (N_OPS * NV);

  if (hypercube_index > n_new_hypercubes - 1)
    return;

  if (hypercube_data_d->occupied + n_new_hypercubes > hypercube_data_d->size)
  {
    if (i == 0)
    {
      printf("Should not have happened: hashmap overflow occured! Decrease hashmap expansion threshold below %f\n",
             (double)hypercube_data_d->occupied / hypercube_data_d->size);
      hypercube_data_d->occupied += n_new_hypercubes;
    }
    return;
  }
  else if (i == 0)
  {
    hypercube_data_d->occupied += n_new_hypercubes;
  }

  const unsigned vector_index = i % (N_OPS * NV);
  const gpu_hashmap_async::hash_key_t k =
      gpu_hashmap_async::key_from_cell<N_DIMS>(new_hypercube_keys_d[hypercube_index]);
  if (insert_vector_element(hypercube_data_d, k,
                            &new_hypercube_data_d[hypercube_index * N_OPS * NV], vector_index))
    return;
}
