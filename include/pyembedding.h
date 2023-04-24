//
// `Damo-Embedding` - 'c++ tool for sparse parameter server'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
//
// This file is part of `Damo-Embedding`.
//
// `Damo-Embedding` is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// `Damo-Embedding` is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with `Damo-Embedding`.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef DAMO_EMBEDDING_PY_EMBEDDING_H
#define DAMO_EMBEDDING_PY_EMBEDDING_H

#pragma once

#include "counting_bloom_filter.h"
#include "cpptoml.h"
#include "embedding.h"
#include "initializer.h"
#include "optimizer.h"

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
  std::string to_string();

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

  /**
   * @brief initialize the weights
   *
   * @param w weights to be initialized
   * @param wn width of the weights
   */
  void call(float *w, int wn);
  ~PyInitializer();

 private:
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

  /**
   * @brief call the optimizer, updating the embedding
   *
   * @param w weights
   * @param wn width of the weights
   * @param gds gradients for weights
   * @param gn width of the grad
   * @param global_step global step
   */
  void call(float *w, int wn, float *gds, int gn,
            unsigned long long global_step);
  ~PyOptimizer();

 private:
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
   * @param key key to find
   * @return true in the filter
   * @return false not in the filter
   */
  bool check(unsigned long long key);

  /**
   * @brief add key to the filter
   *
   * @param key key to ad
   * @param num add counts
   */
  void add(unsigned long long key, unsigned long long num);
  ~PyFilter();

 private:
  std::shared_ptr<CountingBloomFilter> filter_;
  friend class PyEmbedding;
};

class PyStorage {
 public:
  PyStorage();

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
   * @param path to save the data
   * @param expires only save the new keys
   */
  void dump(const std::string &path, int expires);

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

  /**
   * @brief lookup the embeddings
   *
   * @param keys keys to lookup
   * @param kn length of the keys
   * @param w weight for the keys
   * @param wn length of the weights
   * @return
   */
  void lookup(unsigned long long *keys, int kn, float *w, int wn);

  /**
   * @brief update the embedding weights
   *
   * @param keys keys to update
   * @param kn length of the keys
   * @param gds gradients for the keys
   * @param gn length of the gradients
   */
  void apply_gradients(unsigned long long *keys, int kn, float *gds, int gn);

 private:
  std::shared_ptr<Embedding> embedding_;
};

#endif  // DAMO_EMBEDDING_PY_EMBEDDING_H