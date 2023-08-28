#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyembedding.h"

namespace py = pybind11;

PYBIND11_MODULE(damo, m) {
  py::class_<Parameters>(m, "Parameters")
      .def(py::init<>())
      .def("insert",
           py::overload_cast<std::string, std::string>(&Parameters::insert))
      .def("insert", py::overload_cast<std::string, int>(&Parameters::insert))
      .def("insert",
           py::overload_cast<std::string, double>(&Parameters::insert))
      .def("insert", py::overload_cast<std::string, bool>(&Parameters::insert))
      .def("__setitem__",
           py::overload_cast<std::string, std::string>(&Parameters::insert))
      .def("__setitem__",
           py::overload_cast<std::string, int>(&Parameters::insert))
      .def("__setitem__",
           py::overload_cast<std::string, double>(&Parameters::insert))
      .def("__setitem__",
           py::overload_cast<std::string, bool>(&Parameters::insert))
      .def("__repr__",
           [](Parameters &p) -> std::string { return p.to_json(); });

  py::class_<PyInitializer>(m, "PyInitializer")
      .def(py::init<>())
      .def(py::init<Parameters>())
      .def("call", &PyInitializer::call)
      .def("__call__", &PyInitializer::call)
      .def("__repr__", [](PyInitializer &p) -> std::string {
        return "<damo.PyInitializer: " + p.initializer_->to_string() + ">";
      });

  py::class_<PyOptimizer>(m, "PyOptimizer")
      .def(py::init<>())
      .def(py::init<Parameters>())
      .def("call", &PyOptimizer::call)
      .def("__call__", &PyOptimizer::call)
      .def("__repr__", [](PyOptimizer &o) -> std::string {
        return "<damo.PyOptimizer: " + o.optimizer_->to_string() + ">";
      });

  py::class_<PyFilter>(m, "PyFilter")
      .def(py::init<>())
      .def(py::init<Parameters>())
      .def("check", &PyFilter::check)
      .def("add", &PyFilter::add)
      .def("__repr__",
           [](PyFilter &f) -> std::string { return "<damo.PyFilter>"; });

  py::class_<PyStorage>(m, "PyStorage")
      .def(py::init<const std::string &, int>())
      .def("dump",
           py::overload_cast<const std::string &, Parameters>(&PyStorage::dump))
      .def("dump", py::overload_cast<const std::string &>(&PyStorage::dump))
      .def("checkpoint", &PyStorage::checkpoint)
      .def("load_from_checkpoint", &PyStorage::load_from_checkpoint)
      .def("__repr__",
           [](PyStorage &s) -> std::string { return "<damo.PyStorage>"; });

  py::class_<PyEmbedding>(m, "PyEmbedding")
      .def(py::init<PyStorage, PyOptimizer, PyInitializer, int, int>())
      .def("lookup", &PyEmbedding::lookup)
      .def("__call__", &PyEmbedding::lookup)
      .def("apply_gradients", &PyEmbedding::apply_gradients)
      .def("step", &PyEmbedding::apply_gradients)
      .def("__repr__",
           [](PyEmbedding &e) -> std::string { return "<damo.PyEmbedding>"; });
}