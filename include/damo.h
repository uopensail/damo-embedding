//
// `Damo-Embedding` - 'c++ tool for sparse parameter server'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// `Damo-Embedding` is provided under: GNU Affero General Public License
// (AGPL3.0) https://www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//

#ifndef DAMO_EMBEDDING_PY_H
#define DAMO_EMBEDDING_PY_H

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "embedding.h"
namespace py = pybind11;

class PyDamo {
public:
  PyDamo() = delete;
  explicit PyDamo(const std::string &config_file);
  ~PyDamo() {}
  void dump(const std::string &dir);
  void checkpoint(const std::string &dir);
  // void load(const std::string &dir);
  void pull(int group, py::array_t<int64_t> keys, py::array_t<float> w);
  void push(uint64_t step_control, int group, py::array_t<int64_t> keys, py::array_t<float> gds);
  std::string to_json();

private:
  std::shared_ptr<EmbeddingWareHouse> warehouse_;
};

#endif // DAMO_EMBEDDING_PY_H