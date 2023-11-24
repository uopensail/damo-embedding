#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "damo.h"

namespace py = pybind11;

PYBIND11_MODULE(damo, m) {
  m.doc() = "damo embedding server";
  m.def("open", &damo_open, "damo embedding open rocksdb");
  m.def("close", &damo_close, "damo embedding close rocksdb");
  m.def("pull", &damo_pull, "damo embedding pull weights");
  m.def("push", &damo_push, "damo embedding push gradients");
  m.def("load", &damo_load, "damo embedding server load from checkpoint");
  m.def("dump", &damo_dump, "damo embedding server dump for inference");
  m.def("checkpoint", &damo_checkpoint, "damo embedding server do checkpoint");
  m.def("embedding", &damo_new, "damo embedding create embedding");
}