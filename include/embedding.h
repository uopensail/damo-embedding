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

#ifndef DAMO_EMBEDDING_EMBEDDING_H
#define DAMO_EMBEDDING_EMBEDDING_H

#pragma once

#include <rocksdb/db.h>
#include <rocksdb/write_batch.h>

#include <mutex>

#include "count_bloom_filter.h"
#include "initializer.h"
#include "optimizer.h"

//最大的锁的个数
#define max_lock_num 16

typedef struct EmbeddingMeta {
  int dim;
  u_int64_t group;
} EmbeddingMeta;

class Embeddings {
private:
  rocksdb::DB *db_;
  u_int64_t lag_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Initializer> initializer_;
  std::shared_ptr<CountBloomFilter> filter_;
  std::mutex lockers_[max_lock_num];
  EmbeddingMeta metas_[max_group];

private:
  std::string *create(u_int64_t &key);
  void update(u_int64_t &key, MetaData *ptr, Float *gds, u_int64_t global_step);

public:
  /**
   * @brief Construct a new Embedding object
   *
   * @param step_lag 最大的滞后步数
   * @param data_dir 数据存放的路径
   * @param optimizer 优化算子
   * @param initializer 初始化算子
   * @param filter 频控
   */
  Embeddings(u_int64_t step_lag, std::string data_dir,
             const std::shared_ptr<Optimizer> &optimizer,
             const std::shared_ptr<Initializer> &initializer,
             const std::shared_ptr<CountBloomFilter> &filter);

  void add_group(int group, int dim);
  ~Embeddings();

  /**
   * @brief 查找
   *
   * @param keys 所要查找的keys
   * @param len 长度
   * @param data 返回的数据
   * @param n 返回的长度
   * @return u_int64_t 全局的step
   */
  u_int64_t lookup(u_int64_t *keys, int len, Float *data, int n);

  /**
   * @brief 更新梯度
   *
   * @param keys 所要更新的keys
   * @param len 长度
   * @param gds 梯度
   * @param n 长度
   * @param global_steps 全局的step，太滞后的step就不会更新
   */
  void apply_gradients(u_int64_t *keys, int len, Float *gds, int n,
                       u_int64_t global_steps);
  void dump(std::string path, int expires);
};

#endif // DAMO_EMBEDDING_EMBEDDING_H