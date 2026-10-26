#include "nucleus_of_strain_cuda.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- helper: check CUDA errors ----------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------- device helper functions ----------

__device__ inline double vec_norm_d(double Y, double X, double Z)
{
    return sqrt(Y * Y + X * X + Z * Z);
}

__device__ inline double safe_atan2_d(double numerator, double denominator)
{
    return atan(numerator / denominator);
}

__device__ inline double safe_log_d(double argument)
{
    return log(argument);
}

__device__ inline double sub_kernel_d_x1_d(double Y, double X, double Z, double rho)
{
    return Y * safe_log_d(Z + rho) + Z * safe_log_d(Y + rho) - X * safe_atan2_d(Y * Z, X * rho);
}

__device__ inline double sub_kernel_d_xz_d(double Y, double zp, double rho)
{
    return 2.0 * zp * safe_log_d(Y + rho);
}

__device__ inline double pow_1_d(int i, int j, int k)
{
    return ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
}

// ---------- main CUDA kernel ----------
// One thread per evaluation point; each thread loops over all prisms.

__global__ void compute_geomech_kernel(
    const double* __restrict__ coordinates,  // [3 * n_eval_points]  y,x,z interleaved
    const double* __restrict__ prisms,       // [6 * n_prisms]       y1,y2,x1,x2,z2,z1
    const double* __restrict__ pressure,     // [n_prisms]
    const double* __restrict__ temperature,  // [n_prisms]
    double mult_p,
    double mult_t,
    double poisson_coeff,
    int n_prisms,
    int n_eval_points,
    double* __restrict__ ux_p,
    double* __restrict__ uy_p,
    double* __restrict__ uz_p,
    double* __restrict__ ux_t,
    double* __restrict__ uy_t,
    double* __restrict__ uz_t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_eval_points) return;

    double yp = coordinates[3 * idx + 0];
    double xp = coordinates[3 * idx + 1];
    double zp_coord = coordinates[3 * idx + 2];

    double acc_ux_p = 0.0;
    double acc_uy_p = 0.0;
    double acc_uz_p = 0.0;
    double acc_ux_t = 0.0;
    double acc_uy_t = 0.0;
    double acc_uz_t = 0.0;

    for (int m = 0; m < n_prisms; m++) {
        const double* prism = &prisms[6 * m];
        double c_z = 0.5 * (prism[4] + prism[5]);

        double sum_krn_x = 0.0, sum_krn_y = 0.0, sum_krn_z = 0.0;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    double y_prism = prism[1 - i];
                    double x_prism = prism[3 - j];
                    double z_prism = prism[5 - k];

                    double Yv  = yp - y_prism;
                    double Xv  = xp - x_prism;
                    double Zv  = zp_coord - z_prism;
                    double Z2v = Zv + 2.0 * c_z;

                    // Reuse Yv^2+Xv^2 for both rho computations
                    double YX2  = Yv * Yv + Xv * Xv;
                    double rho  = sqrt(YX2 + Zv  * Zv);
                    double rho2 = sqrt(YX2 + Z2v * Z2v);

                    // Precompute 6 unique log terms (avoids 6 redundant log calls)
                    double logYrho   = safe_log_d(Yv  + rho);
                    double logZrho   = safe_log_d(Zv  + rho);
                    double logXrho   = safe_log_d(Xv  + rho);
                    double logYrho2  = safe_log_d(Yv  + rho2);
                    double logZ2rho2 = safe_log_d(Z2v + rho2);
                    double logXrho2  = safe_log_d(Xv  + rho2);

                    // Shared atan term reused by z2 (k7) and zz2 (k8)
                    double atan_z2 = safe_atan2_d(Xv * Yv, Z2v * rho2);

                    double k0 = Yv * logZrho   + Zv  * logYrho   - Xv * safe_atan2_d(Yv * Zv,  Xv * rho);   // x1
                    double k1 = Yv * logZ2rho2 + Z2v * logYrho2  - Xv * safe_atan2_d(Yv * Z2v, Xv * rho2);  // x2
                    double k3 = Xv * logZrho   + Zv  * logXrho   - Yv * safe_atan2_d(Xv * Zv,  Yv * rho);   // y1
                    double k4 = Xv * logZ2rho2 + Z2v * logXrho2  - Yv * safe_atan2_d(Xv * Z2v, Yv * rho2);  // y2
                    double k6 = Xv * logYrho   + Yv  * logXrho   - Zv * safe_atan2_d(Xv * Yv,  Zv * rho);   // z1
                    double k7 = Xv * logYrho2  + Yv  * logXrho2  - Z2v * atan_z2;                             // z2
                    double k2 = 2.0 * zp_coord * logYrho2;   // xz2
                    double k5 = 2.0 * zp_coord * logXrho2;   // yz2
                    double k8 = -2.0 * zp_coord * atan_z2;   // zz2

                    double krn_x = k0 + poisson_coeff * k1 + k2;
                    double krn_y = k3 + poisson_coeff * k4 + k5;
                    double krn_z = k6 - poisson_coeff * k7 + k8;

                    double pow_ = pow_1_d(i, j, k);
                    sum_krn_x += pow_ * krn_x;
                    sum_krn_y += pow_ * krn_y;
                    sum_krn_z += pow_ * krn_z;
                }
            }
        }

        acc_ux_p += pressure[m]    * sum_krn_x;
        acc_uy_p += pressure[m]    * sum_krn_y;
        acc_uz_p += pressure[m]    * sum_krn_z;
        acc_ux_t += temperature[m] * sum_krn_x;
        acc_uy_t += temperature[m] * sum_krn_y;
        acc_uz_t += temperature[m] * sum_krn_z;
    }

    ux_p[idx] = acc_ux_p * mult_p;
    uy_p[idx] = acc_uy_p * mult_p;
    uz_p[idx] = acc_uz_p * mult_p;
    ux_t[idx] = acc_ux_t * mult_t;
    uy_t[idx] = acc_uy_t * mult_t;
    uz_t[idx] = acc_uz_t * mult_t;
}


// ---------- host wrapper ----------

// Uniaxial compaction coefficient
static inline double Cm_host(double poisson, double young)
{
    return ((1.0 + poisson) * (1.0 - 2.0 * poisson)) / (young * (1.0 - poisson));
}

std::map<std::string, std::vector<double>>
compute_geomech_cuda(
    std::vector<double>& coordinates,
    std::vector<double>& prisms,
    std::vector<double>& pressure,
    double poisson,
    double young,
    std::vector<double>& temperature,
    double thermal_expansion)
{
    int n_eval_points = (int)(coordinates.size() / 3);
    int n_prisms      = (int)(prisms.size() / 6);

    double mult_p = -Cm_host(poisson, young) / (4.0 * M_PI);
    double mult_t = -thermal_expansion * (1.0 + poisson) / (1.0 - poisson) / (4.0 * M_PI);
    double poisson_coeff = 3.0 - 4.0 * poisson;

    // --- allocate device memory ---
    double *d_coords, *d_prisms, *d_pressure, *d_temperature;
    double *d_ux_p, *d_uy_p, *d_uz_p;
    double *d_ux_t, *d_uy_t, *d_uz_t;

    CUDA_CHECK(cudaMalloc(&d_coords,      coordinates.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prisms,      prisms.size()      * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pressure,    n_prisms           * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temperature, n_prisms           * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_ux_p, n_eval_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_uy_p, n_eval_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_uz_p, n_eval_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ux_t, n_eval_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_uy_t, n_eval_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_uz_t, n_eval_points * sizeof(double)));

    // --- copy inputs to device ---
    CUDA_CHECK(cudaMemcpy(d_coords,      coordinates.data(), coordinates.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prisms,      prisms.data(),      prisms.size()      * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pressure,    pressure.data(),    n_prisms           * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temperature, temperature.data(), n_prisms           * sizeof(double), cudaMemcpyHostToDevice));

    // --- launch kernel ---
    int blockSize = 256;
    int gridSize  = (n_eval_points + blockSize - 1) / blockSize;

    compute_geomech_kernel<<<gridSize, blockSize>>>(
        d_coords, d_prisms, d_pressure, d_temperature,
        mult_p, mult_t, poisson_coeff,
        n_prisms, n_eval_points,
        d_ux_p, d_uy_p, d_uz_p,
        d_ux_t, d_uy_t, d_uz_t);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- copy results back ---
    std::vector<double> ux_p(n_eval_points), uy_p(n_eval_points), uz_p(n_eval_points);
    std::vector<double> ux_t(n_eval_points), uy_t(n_eval_points), uz_t(n_eval_points);

    CUDA_CHECK(cudaMemcpy(ux_p.data(), d_ux_p, n_eval_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy_p.data(), d_uy_p, n_eval_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uz_p.data(), d_uz_p, n_eval_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ux_t.data(), d_ux_t, n_eval_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy_t.data(), d_uy_t, n_eval_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uz_t.data(), d_uz_t, n_eval_points * sizeof(double), cudaMemcpyDeviceToHost));

    // --- free device memory ---
    cudaFree(d_coords);
    cudaFree(d_prisms);
    cudaFree(d_pressure);
    cudaFree(d_temperature);
    cudaFree(d_ux_p);
    cudaFree(d_uy_p);
    cudaFree(d_uz_p);
    cudaFree(d_ux_t);
    cudaFree(d_uy_t);
    cudaFree(d_uz_t);

    // --- build result map (same format as CPU version) ---
    std::map<std::string, std::vector<double>> result;
    result["ux_p"] = std::move(ux_p);
    result["uy_p"] = std::move(uy_p);
    result["uz_p"] = std::move(uz_p);
    result["ux_t"] = std::move(ux_t);
    result["uy_t"] = std::move(uy_t);
    result["uz_t"] = std::move(uz_t);

    return result;
}
