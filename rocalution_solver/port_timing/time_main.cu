#include "port_bench.h"

#include <cstdio>

#include <hip/hip_runtime.h>

static int run_one(const char *name, int n, int (*fn)(int, int, int, double *, double *))
{
  const int warmup = 10;
  const int samples = 100;
  double median_ms = 0;
  double checksum = 0;
  const int rc = fn(n, warmup, samples, &median_ms, &checksum);
  if (rc)
  {
    std::printf("FAIL %s %d\n", name, n);
    return rc;
  }
  std::printf("MEDIAN %s %d %.6f %.17g\n", name, n, median_ms, checksum);
  return 0;
}

int main()
{
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess)
  {
    std::printf("FAIL device\n");
    return 1;
  }
  std::printf("WARMUP 10\n");
  std::printf("SAMPLES 100\n");
  std::printf("BACKEND hip\n");
  std::printf("DEVICE %s\n", prop.name);

  const int sizes[2] = {100, 1 << 20};
  const char *names[4] = {"residual", "cpr", "schur", "brp"};
  int (*fns[4])(int, int, int, double *, double *) = {
      od_bench_residual, od_bench_cpr, od_bench_schur, od_bench_brp};
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 4; ++k)
      if (run_one(names[k], sizes[s], fns[k]))
        return 1;

  std::printf("PORT_TIME_OK\n");
  return 0;
}
