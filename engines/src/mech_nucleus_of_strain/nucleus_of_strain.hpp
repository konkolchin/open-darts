#pragma once

#include <math.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <array>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef DBL_EPSILON
#define DBL_EPSILON 1e-15
#endif

inline double vec_norm(double Y, double X, double Z)
{
    return sqrt(Y*Y + X*X + Z*Z);
}

inline double safe_atan2(double numerator, double denominator)
{
/*
Principal value of the arctangent expressed as a two variable function

This modification has to be made to the arctangent function so the
harmonic field of the prism satisfies the Poisson's equation.
Therefore, it guarantees that the fields satisfies the symmetry properties
of the prism.This modified function has been defined according to
Fukushima(2019).
*/
    double result;

    //if (fabs(denominator) > DBL_EPSILON) {
        result = atan(numerator / denominator);
    /* }
    else {
        if (numerator > 0)
            result = M_PI / 2;
        else if (numerator < 0)
            result = -M_PI / 2;
        else
            result = 0;
    }*/
    return result;
}


inline double safe_log(double argument)
{
    /*
        Modified log to return 0 for log(0).
        The limits in the formula terms tend to 0 (see Nagy et al., 2000).
    */
    double result;
    //if (fabs(argument) < 1e-10)
    //    result = 0;
    //else
        result = log(argument);
    return result;
}

// Uniaxial compaction coefficient Cm.
inline double Cm(double poisson, double young)
{
    double result;
    result = ((1 + poisson) * (1 - 2 * poisson)) / (young * (1 - poisson));
    return result;
}

inline double sub_kernel_d_x1(double Y, double X, double Z, double rho)
{
    return Y * safe_log(Z + rho) + Z * safe_log(Y + rho) - X * safe_atan2(Y * Z, X * rho);
}

inline double sub_kernel_d_xz(double Y, double zp, double rho)
{
    return 2 * zp * safe_log(Y + rho);
}


inline double pow_1(int i, int j, int k)
{
#if 0
    return pow(-1, (i + j + k));
#else
    if ((i + j + k) % 2 == 0)
        return 1;
    else
        return -1;
#endif
}


std::array<double, 6>
compute_point(std::array<double, 3>& coordinates, std::vector<double>& prisms, std::vector<double>& pressure,
    std::vector<double>& temperature, double poisson);

std::array<double, 9>
kernel_displs(double y, double x, double z, double zc, double yp, double xp, double zp);

std::map<std::string, std::vector<double>>
compute_geomech(
    std::vector<double>& coordinates,
    std::vector<double>& prisms,
    std::vector<double>& pressure,
    double poisson,
    double young,
    std::vector<double>& temperature,
    double thermal_expansion);

void set_num_threads(int n);
void set_platform(const std::string& platform);
std::string get_platform();

#ifdef USE_CUDA
#include "nucleus_of_strain_cuda.hpp"
#endif
