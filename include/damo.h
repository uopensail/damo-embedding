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

#include "embedding.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

const std::string default_dir = "./embeddings";
const int default_ttl = 86400 * 30;

void damo_open(int ttl = default_ttl, const std::string &dir = default_dir);
void damo_close();
void damo_new(const std::string &params);
void damo_dump(const std::string &dir);
void damo_checkpoint(const std::string &dir);
void damo_load(const std::string &dir);
void damo_pull(int group, py::array_t<int64_t> keys, py::array_t<float> w);
void damo_push(int group, py::array_t<int64_t> keys, py::array_t<float> gds);

#endif // DAMO_EMBEDDING_PY_H