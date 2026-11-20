#pragma once

// One kernel from each ported translation unit.
// median_ms is the median device-event time of `samples` launches after `warmup`.
// checksum is taken after those launches, from the last kernel output.

#ifdef __cplusplus
extern "C" {
#endif

int od_bench_residual(int n, int warmup, int samples, double *median_ms, double *checksum);
int od_bench_cpr(int n, int warmup, int samples, double *median_ms, double *checksum);
int od_bench_schur(int n, int warmup, int samples, double *median_ms, double *checksum);
int od_bench_brp(int n, int warmup, int samples, double *median_ms, double *checksum);

#ifdef __cplusplus
}
#endif
