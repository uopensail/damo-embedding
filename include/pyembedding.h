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

#include "count_bloom_filter.h"
#include "cpptoml.h"
#include "embedding.h"
#include "initializer.h"
#include "optimizer.h"

class Parameters {
public:
  std::shared_ptr<cpptoml::table> params_;

public:
  Parameters();
  Parameters(const Parameters &p);
  Parameters &operator=(const Parameters &p);
  ~Parameters();
  void insert(std::string key, std::string value);
  void insert(std::string key, int value);
  void insert(std::string key, double value);
  void insert(std::string key, bool value);
};

class PyEmbedding;

class PyInitializer {
private:
  std::shared_ptr<Initializer> initializer_;
  friend class PyEmbedding;

public:
  PyInitializer();
  PyInitializer(Parameters params);
  PyInitializer(const PyInitializer &p);
  PyInitializer &operator=(const PyInitializer &p);
  void call(float *w, int wn);
  ~PyInitializer();
};

class PyOptimizer {
private:
  std::shared_ptr<Optimizer> optimizer_;
  friend class PyEmbedding;

public:
  PyOptimizer();
  PyOptimizer(Parameters op_params);
  PyOptimizer(Parameters op_params, Parameters decay_params);
  PyOptimizer(const PyOptimizer &p);
  PyOptimizer &operator=(const PyOptimizer &p);
  void call(float *w, int wn, float *gds, int gn,
            unsigned long long global_step);
  ~PyOptimizer();
};

class PyFilter {
private:
  std::shared_ptr<CountBloomFilter> filter_;
  friend class PyEmbedding;

public:
  PyFilter();
  PyFilter(Parameters params);
  PyFilter(const PyFilter &p);
  PyFilter &operator=(const PyFilter &p);
  bool check(unsigned long long key);
  void add(unsigned long long key, unsigned long long num);
  ~PyFilter();
};

class PyEmbedding {
private:
  std::shared_ptr<Embedding> embedding_;

public:
  PyEmbedding() = delete;
  PyEmbedding(const PyEmbedding &p) = delete;
  PyFilter &operator=(const PyFilter &p) = delete;

public:
  /**
   * @brief Construct a new Py Embedding object
   *
   * @param dim 维度
   * @param max_lag 最大滞后步数
   * @param data_dir 数据存放的路径
   * @param filter  频控, 可为空
   * @param optimizer 优化算子
   * @param initializer 初始化算子
   */
  PyEmbedding(int gid, int dim, unsigned long long max_lag, std::string data_dir,
              PyFilter filter, PyOptimizer optimizer,
              PyInitializer initializer);
  ~PyEmbedding();

  /**
   * @brief 查询
   *
   * @param keys 需要查询的keys
   * @param kn keys的长度
   * @param w 返回的数据
   * @param wn 返回的数据长度
   * @return unsigned long long
   */
  unsigned long long lookup(unsigned long long *keys, int kn, float *w, int wn);

  /**
   * @brief
   *
   * @param keys 需要更新的keys
   * @param kn keys的长度
   * @param gds 梯度
   * @param gn 梯度权重的长度
   * @param global_step 全局step
   */
  void apply_gradients(unsigned long long *keys, int kn, float *gds, int gn,
                       unsigned long long global_step);

  /**
   * @brief 保存权重到磁盘
   *
   * @param path 路径
   * @param expires out of days, 过期天数
   */
  void dump(std::string path, int expires);
};

#endif // DAMO_EMBEDDING_PY_EMBEDDING_H