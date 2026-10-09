#pragma once

#include <vector>
#include <map>
#include <string>

// CUDA-accelerated version of compute_geomech
// Returns the same result map as the CPU version
std::map<std::string, std::vector<double>>
compute_geomech_cuda(
    std::vector<double>& coordinates,
    std::vector<double>& prisms,
    std::vector<double>& pressure,
    double poisson,
    double young,
    std::vector<double>& temperature,
    double thermal_expansion);
