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

#ifndef DAMO_EMBEDDING_PY_EMBEDDING_H
#define DAMO_EMBEDDING_PY_EMBEDDING_H

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "counting_bloom_filter.h"
#include "cpptoml.h"
#include "embedding.h"
#include "initializer.h"
#include "optimizer.h"

namespace py = pybind11;

class Parameters {
 public:
  Parameters();
  Parameters(const Parameters &p);
  Parameters &operator=(const Parameters &p);
  ~Parameters();
  void insert(std::string key, std::string value);
  void insert(std::string key, int value);
  void insert(std::string key, double value);
  void insert(std::string key, bool value);
  std::string to_json();

 public:
  std::shared_ptr<cpptoml::table> params_;
};

class PyEmbedding;

class PyInitializer {
 public:
  PyInitializer();
  PyInitializer(Parameters params);
  PyInitializer(const PyInitializer &p);
  PyInitializer &operator=(const PyInitializer &p);
  void call(py::array_t<float> w);
  ~PyInitializer();

 public:
  std::shared_ptr<Initializer> initializer_;
  friend class PyEmbedding;
};

class PyOptimizer {
 public:
  PyOptimizer();
  PyOptimizer(Parameters op_params);
  PyOptimizer(Parameters op_params, Parameters decay_params);
  PyOptimizer(const PyOptimizer &p);
  PyOptimizer &operator=(const PyOptimizer &p);
  void call(py::array_t<float> w, py::array_t<float> gd,
            int64_t global_step = 0);

  ~PyOptimizer();

 public:
  std::shared_ptr<Optimizer> optimizer_;
  friend class PyEmbedding;
};

class PyFilter {
 public:
  PyFilter();
  PyFilter(Parameters params);
  PyFilter(const PyFilter &p);
  PyFilter &operator=(const PyFilter &p);

  /**
   * @brief where the key in the filter
   *
   * @param group key group
   * @param key key to find
   * @return true in the filter
   * @return false not in the filter
   */
  bool check(int group, int64_t key);

  /**
   * @brief add key to the filter
   *
   * @param group key group
   * @param key key to ad
   * @param num add counts
   */
  void add(int group, int64_t key, int64_t num);
  ~PyFilter();

 private:
  std::shared_ptr<CountingBloomFilter> filter_;
  friend class PyEmbedding;
};

class PyStorage {
 public:
  /**
   * @brief Construct a new Py Storage object
   *
   * @param data_dir to save the data
   * @param ttl time to live in seconds
   */
  PyStorage(const std::string &data_dir, int ttl = 0);
  ~PyStorage();

  /**
   * @brief dump the data to binary format for online predict
   *
   * @param path data path to dump
   * @param condition condition for dump
   */
  void dump(const std::string &path, Parameters condition);

  /**
   * @brief dump the data to binary format for online predict
   *
   * @param path data path to dump
   */
  void dump(const std::string &path);

  /**
   * @brief do the checkpoint
   *
   * @param path file path
   */
  void checkpoint(const std::string &path);

  /**
   * @brief load from checkpoint file
   *
   * @param path file path
   */
  void load_from_checkpoint(const std::string &path);

 private:
  std::shared_ptr<Storage> storage_;
  friend class PyEmbedding;
};

class PyEmbedding {
 public:
  PyEmbedding() = delete;
  PyEmbedding(PyStorage storage, PyOptimizer optimizer,
              PyInitializer initializer, int dim, int group = 0);
  PyEmbedding(const PyEmbedding &p);
  PyEmbedding &operator=(const PyEmbedding &p);
  ~PyEmbedding();

  void lookup(py::array_t<int64_t> keys, py::array_t<float> w);
  void apply_gradients(py::array_t<int64_t> keys, py::array_t<float> gds);

 private:
  std::shared_ptr<Embedding> embedding_;
};

#endif  // DAMO_EMBEDDING_PY_EMBEDDING_H