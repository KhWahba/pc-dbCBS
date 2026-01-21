#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "pc_dbcbs_api.hpp"
#include <csignal>
#include <cstdlib>

namespace py = pybind11;

PYBIND11_MODULE(pcdbcbs, m) {
    std::signal(SIGINT, [](int) { std::quick_exit(130);});

  m.doc() = "Python bindings for pc-dbCBS (pcdbcbs_api)";

  py::class_<pcdbcbs::Options>(m, "Options")
      .def(py::init<>())
      .def_readwrite("input_yaml", &pcdbcbs::Options::input_yaml)
      .def_readwrite("output_yaml", &pcdbcbs::Options::output_yaml)
      .def_readwrite("optimization_yaml", &pcdbcbs::Options::optimization_yaml)
      .def_readwrite("pc_dbcbs_cfg_yaml", &pcdbcbs::Options::pc_dbcbs_cfg_yaml)
      .def_readwrite("opt_cfg_yaml", &pcdbcbs::Options::opt_cfg_yaml)
      .def_readwrite("time_limit", &pcdbcbs::Options::time_limit)
      .def_readwrite("override_visualize_mujoco", &pcdbcbs::Options::override_visualize_mujoco)
      .def_readwrite("visualize_mujoco", &pcdbcbs::Options::visualize_mujoco)
      .def_readwrite("dynobench_base", &pcdbcbs::Options::dynobench_base)
      .def_readwrite("motion_primitives_base", &pcdbcbs::Options::motion_primitives_base);

  py::class_<pcdbcbs::Result>(m, "Result")
      .def_readonly("solved_db", &pcdbcbs::Result::solved_db)
      .def_readonly("solved_opt", &pcdbcbs::Result::solved_opt)
      .def_readonly("duration_discrete_sec", &pcdbcbs::Result::duration_discrete_sec)
      .def_readonly("duration_opt_sec", &pcdbcbs::Result::duration_opt_sec)
      .def_readonly("cost", &pcdbcbs::Result::cost)
      .def_readonly("feasible", &pcdbcbs::Result::feasible)
      .def_readonly("info", &pcdbcbs::Result::info)
      .def_readonly("X", &pcdbcbs::Result::X)   // numpy array via pybind11/eigen.h
      .def_readonly("U", &pcdbcbs::Result::U);

  m.def(
      "run",
      static_cast<pcdbcbs::Result (*)(const pcdbcbs::Options&)>(&pcdbcbs::run),
      py::arg("opt"),
      "Run pc-dbCBS (discrete + optimization) and return Result");

}
