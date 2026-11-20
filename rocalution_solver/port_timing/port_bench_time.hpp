#pragma once

// Host-side HIP event timing for one kernel launch.

#include <algorithm>
#include <vector>

#include <hip/hip_runtime.h>

template <typename Launch>
int od_port_time(int warmup, int samples, Launch launch, double *median_ms)
{
  for (int i = 0; i < warmup; ++i)
  {
    launch();
    if (hipGetLastError() != hipSuccess)
      return 1;
  }
  if (hipDeviceSynchronize() != hipSuccess)
    return 1;

  hipEvent_t start, stop;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess)
    return 1;

  std::vector<float> ms((size_t)samples);
  for (int i = 0; i < samples; ++i)
  {
    if (hipEventRecord(start) != hipSuccess)
      return 1;
    launch();
    if (hipEventRecord(stop) != hipSuccess)
      return 1;
    if (hipEventSynchronize(stop) != hipSuccess)
      return 1;
    if (hipGetLastError() != hipSuccess)
      return 1;
    if (hipEventElapsedTime(&ms[(size_t)i], start, stop) != hipSuccess)
      return 1;
  }
  hipEventDestroy(start);
  hipEventDestroy(stop);

  std::sort(ms.begin(), ms.end());
  if (samples % 2)
    *median_ms = (double)ms[(size_t)samples / 2];
  else
    *median_ms = 0.5 * ((double)ms[(size_t)samples / 2 - 1] + (double)ms[(size_t)samples / 2]);
  return 0;
}
