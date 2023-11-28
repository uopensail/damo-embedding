#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "damo.h"

namespace py = pybind11;

PYBIND11_MODULE(damo, m) {
  m.doc() = "damo embedding";
  py::class_<PyDamo>(m, "PyDamo")
      .def(py::init<const std::string &>())
      .def("pull", &PyDamo::pull)
      .def("push", &PyDamo::push)
      //.def("load", &PyDamo::load)
      .def("dump", &PyDamo::dump)
      .def("checkpoint", &PyDamo::checkpoint)
      .def("__repr__", [](PyDamo &p) -> std::string { return p.to_json(); });
}