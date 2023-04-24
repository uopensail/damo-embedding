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
#include <rocksdb/utilities/db_ttl.h>
#include <rocksdb/write_batch.h>

#include "initializer.h"
#include "optimizer.h"

class Embedding;
class Storage {
 public:
  Storage() = delete;
  Storage(int ttl, const std::string &data_dir);
  ~Storage();

  /**
   * @brief save the data to filesystem, with the given filter condition
   *
   * @param path file path
   * @param filter condition
   */
  void dump(const std::string &path,
            const std::function<bool(MetaData *ptr)> &filter);

 private:
  int ttl_;
  bool groups_[max_group];
  std::shared_ptr<rocksdb::DBWithTTL> db_;
  friend class Embedding;
};

class Embedding {
 public:
  Embedding() = delete;
  Embedding(Storage &storage, const std::shared_ptr<Optimizer> &optimizer,
            const std::shared_ptr<Initializer> &initializer, int dim,
            int min_count, int group = 0);
  ~Embedding();
  /**
   * @brief lookup the embeddings
   *
   * @param keys keys to lookup
   * @param kn length of the keys
   * @param w weight for the keys
   * @param wn length of the weights
   * @return
   */
  void lookup(u_int64_t *keys, int len, Float *data, int n);

  /**
   * @brief update the embedding weights
   *
   * @param keys keys to update
   * @param kn length of the keys
   * @param gds gradients for the keys
   * @param gn length of the gradients
   * @param global_step global step
   */
  void apply_gradients(u_int64_t *keys, int len, Float *gds, int n,
                       const u_int64_t &global_step);
  const int get_dim() const;
  const u_int64_t get_min_count() const;

 private:
  void update(const u_int64_t &key, MetaData *ptr, Float *gds,
              const u_int64_t &global_step);
  void update(const u_int64_t &key, MetaData *ptr);
  std::shared_ptr<std::string> create(const u_int64_t &key);

 private:
  int dim_;
  int group_;
  u_int64_t group_mask_;
  u_int64_t min_count_;
  std::shared_ptr<rocksdb::DBWithTTL> db_;
  const std::shared_ptr<Optimizer> optimizer_;
  const std::shared_ptr<Initializer> initializer_;
};

#endif  // DAMO_EMBEDDING_EMBEDDING_H