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

class Embedding {
 private:
  rocksdb::DB *db_;
  int dim_;
  u_int64_t lag_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Initializer> initializer_;
  std::shared_ptr<CountBloomFilter> filter_;
  std::mutex lockers_[max_lock_num];

 private:
  void create(u_int64_t &key, std::string &value);
  void update(MetaData *ptr, Float *gds, u_int64_t global_step);

 public:
  Embedding(int dim, u_int64_t step_lag, std::string data_dir,
            const std::shared_ptr<Optimizer> &optimizer,
            const std::shared_ptr<Initializer> &initializer,
            const std::shared_ptr<CountBloomFilter> &filter);
  ~Embedding();
  u_int64_t lookup(u_int64_t *keys, int len, Float *data, int n);
  void apply_gradients(u_int64_t *keys, int len, Float *gds, int n,
                       u_int64_t global_steps);
  void dump(std::string path, int expires);
};

#endif  // DAMO_EMBEDDING_EMBEDDING_H