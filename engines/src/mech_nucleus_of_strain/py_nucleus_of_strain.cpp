#include "py_global.h"
#include "nucleus_of_strain.hpp"

namespace py = pybind11;

// One module for CPU and GPU builds: a GPU build adds compute_geomech_cuda and lets
// set_platform("gpu") route compute_geomech to it.
PYBIND11_MODULE(_proxygeomech, m)
{
	py::bind_vector<std::vector<index_t>>(m, "index_vector", py::module_local(), py::buffer_protocol());
	py::bind_vector<std::vector<value_t>>(m, "value_vector", py::module_local(), py::buffer_protocol())
		.def(py::pickle(
			[](const std::vector<value_t> &p) { // __getstate__
		py::tuple t(p.size());
		for (size_t i = 0; i < p.size(); i++)
			t[i] = p[i];

		return t;
	},
			[](py::tuple t) { // __setstate__
		std::vector<value_t> p(t.size());

		for (size_t i = 0; i < p.size(); i++)
			p[i] = t[i].cast<value_t>();

		return p;
	}));

#ifdef USE_CUDA
	m.def("compute_geomech", [](std::vector<double>& coordinates, std::vector<double>& prisms,
	                            std::vector<double>& pressure, double poisson, double young,
	                            std::vector<double>& temperature, double thermal_expansion) {
		if (get_platform() == "gpu")
			return compute_geomech_cuda(coordinates, prisms, pressure, poisson, young, temperature, thermal_expansion);
		return compute_geomech(coordinates, prisms, pressure, poisson, young, temperature, thermal_expansion);
	});
	m.def("compute_geomech_cuda", compute_geomech_cuda);
#else
	m.def("compute_geomech", compute_geomech);
#endif
	m.def("set_num_threads", set_num_threads, py::arg("n"));
	m.def("set_platform", set_platform, py::arg("platform"));
}
