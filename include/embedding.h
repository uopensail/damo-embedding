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

#ifndef DAMO_EMBEDDING_EMBEDDING_H
#define DAMO_EMBEDDING_EMBEDDING_H

#pragma once

#include <rocksdb/db.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/utilities/db_ttl.h>
#include <rocksdb/write_batch.h>

#include <mutex>

#include "initializer.h"
#include "optimizer.h"

typedef struct Configure {
  int dim;
  int group;
  std::shared_ptr<Optimizer> optimizer;
  std::shared_ptr<Initializer> initializer;
  Configure();
} Configure;

class Embedding;
class Storage;

class GlobalGroupConfigure {
 public:
  GlobalGroupConfigure();
  ~GlobalGroupConfigure() = default;
  const Configure *operator[](int group) const;

 private:
  void add(int group, const Configure &configure);

 private:
  std::mutex group_lock_;
  std::shared_ptr<std::unordered_map<int, Configure>> configures_;
  friend class Embedding;
  friend class Storage;
};

static GlobalGroupConfigure global_groiup_configure;

class ApplyGredientsOperator : public rocksdb::MergeOperator {
 public:
  ApplyGredientsOperator() {}
  ~ApplyGredientsOperator() {}

  virtual bool FullMerge(const rocksdb::Slice &key,
                         const rocksdb::Slice *existing_value,
                         const std::deque<std::string> &operand_list,
                         std::string *new_value,
                         rocksdb::Logger *logger) const override;

  virtual bool PartialMerge(const rocksdb::Slice &key,
                            const rocksdb::Slice &left_operand,
                            const rocksdb::Slice &right_operand,
                            std::string *new_value,
                            rocksdb::Logger *logger) const override {
    return false;
  }

  static const char *kClassName() { return "ApplyGredientsOperator"; }
  static const char *kNickName() { return "apply_gredients"; }
  [[nodiscard]] const char *Name() const override { return kClassName(); }
  [[nodiscard]] const char *NickName() const { return kNickName(); }
};

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
  int ttl_;
  std::shared_ptr<rocksdb::DBWithTTL> db_;
  friend class Embedding;
};

class Embedding {
 public:
  Embedding() = delete;
  Embedding(Storage &storage, const std::shared_ptr<Optimizer> &optimizer,
            const std::shared_ptr<Initializer> &initializer, int dim,
            int group = 0);
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
  void lookup(int64_t *keys, int len, Float *data, int n);

  /**
   * @brief update the embedding weights
   *
   * @param keys keys to update
   * @param kn length of the keys
   * @param gds gradients for the keys
   * @param gn length of the gradients
   */
  void apply_gradients(int64_t *keys, int len, Float *gds, int n);

 private:
  std::shared_ptr<std::string> create(const int64_t &key);

 private:
  int dim_;
  int group_;
  std::shared_ptr<rocksdb::DBWithTTL> db_;
  const std::shared_ptr<Optimizer> optimizer_;
  const std::shared_ptr<Initializer> initializer_;
};

#endif  // DAMO_EMBEDDING_EMBEDDING_H