#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "engine_super_gpu.hpp"
#include "conn_mesh.h"

template <uint8_t NC, uint8_t NP, bool THERMAL>
struct engine_super_gpu_exposer
{
  static void expose(py::module& m)
  {
    std::string short_name, long_name;
    short_name = "engine_super_gpu" + std::to_string(NC) + "_" + std::to_string(NP);
    if (THERMAL)
    {
      long_name = "Isothermal ";
      short_name += "_t";
    }
    else
    {
      long_name = "Non-isothermal ";
    }
    long_name += "GPU simulator engine for " + std::to_string(NC) + " components and " + std::to_string(NP) + " phases with diffusion and kinetic reaction";
    py::class_<engine_super_gpu<NC, NP, THERMAL>, engine_base>(m, short_name.c_str(), long_name.c_str())
      .def(py::init<>())
      .def("init", (int (engine_super_gpu<NC, NP, THERMAL>::*)(conn_mesh*, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, operator_set_gradient_evaluator_iface*, sim_params*, timer_node*)) & engine_super_gpu<NC, NP, THERMAL>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 6>()) \
      .def("get_X_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(self.X_d, "double_ptr");
          }) \
      // previous-timestep state: push host edits of engine.Xn with copy_data_to_device(engine.Xn, engine.get_Xn_d())
      .def("get_Xn_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::object {
            if (!self.Xn_d) return py::none();
            return py::capsule(&(self.Xn_d), "double_ptr_ptr");
          }) \
      .def("get_RHS_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(self.RHS_d, "double_ptr");
          }) \
      .def("get_molar_weights_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(&(self.molar_weights_d), "double_ptr_ptr");
          }) \
      .def("get_darcy_velocities_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(&(self.darcy_velocities_d), "double_ptr_ptr");
          }) \
      .def("get_velocity_appr_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(&(self.mesh_velocity_appr_d), "double_ptr_ptr");
          }) \
      .def("get_velocity_offset_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(&(self.mesh_velocity_offset_d), "int_ptr_ptr");
          }) \
      .def("get_op_num_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(&(self.mesh_op_num_d), "int_ptr_ptr");
          }) \
      .def("get_dispersivity_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::capsule {
            return py::capsule(&(self.dispersivity_d), "double_ptr_ptr");
          }) \
      // Device copies of mesh.tran / mesh.tranD, made by init() and read at every assembly: after editing mesh.tran on
      // the host (e.g. shutting perforations), push it with copy_data_to_device(mesh.tran, engine.get_mesh_tran_d()).
      // None before init(), which copies the host arrays itself.
      .def("get_mesh_tran_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::object {
            if (!self.mesh_tran_d) return py::none();
            return py::capsule(&(self.mesh_tran_d), "double_ptr_ptr");
          }) \
      .def("get_mesh_tranD_d", [](const engine_super_gpu<NC, NP, THERMAL>& self) -> py::object {
            if (!self.mesh_tranD_d) return py::none();
            return py::capsule(&(self.mesh_tranD_d), "double_ptr_ptr");
          });
  };
};

void pybind_engine_super_gpu(py::module &m)
{
  recursive_exposer_nc_np_t<engine_super_gpu_exposer, py::module, 2, MAX_NC, 1, false> re;
  re.expose(m);

  recursive_exposer_nc_np_t<engine_super_gpu_exposer, py::module, 2, MAX_NC, 2, false> re1;
  re1.expose(m);

  recursive_exposer_nc_np_t<engine_super_gpu_exposer, py::module, 2, MAX_NC, 3, false> re2;
  re2.expose(m);

  recursive_exposer_nc_np_t<engine_super_gpu_exposer, py::module, 1, MAX_NC, 1, true> re3;
  re3.expose(m);

  recursive_exposer_nc_np_t<engine_super_gpu_exposer, py::module, 1, MAX_NC, 2, true> re4;
  re4.expose(m);

  recursive_exposer_nc_np_t<engine_super_gpu_exposer, py::module, 1, MAX_NC, 3, true> re5;
  re5.expose(m);

  // GPU counterpart of the 4-phase thermal CPU engines (engine_super_cpu2_4_t,
  // engine_super_cpu3_4_t): CO2-H2O with aqueous, gaseous CO2, liquid CO2 and
  // hydrate (or ice), optionally with the kinetic hydrate pseudo-component
  recursive_exposer_nc_np_t<engine_super_gpu_exposer, py::module, 2, 3, 4, true> re6;
  re6.expose(m);
}

#endif //PYBIND11_ENABLED
