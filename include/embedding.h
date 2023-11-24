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

// damo embedding supports 256 embeddings
const int max_embedding_num = 256;

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
  [[nodiscard]] const char *NickName() const override { return kNickName(); }
};

class Embedding {
public:
  Embedding() = delete;
  explicit Embedding(json &params);
  ~Embedding();

public:
  json &params;
  int dim;
  int group;
  std::shared_ptr<Optimizer> optimizer;
  std::shared_ptr<Initializer> initializer;
};

class EmbeddingWareHouse {
public:
  EmbeddingWareHouse();
  ~EmbeddingWareHouse();

  void opendb(int ttl, const std::string &data_dir);
  void dump(const std::string &path);
  void checkpoint(const std::string &path);
  void load(const std::string &path);
  void closedb();

  int size() const;
  Embedding *insert(json &params);
  Embedding *operator[](int group) const;

  void lookup(int group, int64_t *keys, int len, Float *data, int n);
  void apply_gradients(int group, int64_t *keys, int len, Float *gds, int n);

private:
  std::shared_ptr<std::string> create_record(int group, const int64_t &key);

private:
  int size_;
  std::mutex lock_;
  Embedding **embeddings_;
  rocksdb::DBWithTTL *db_;
};

static std::shared_ptr<EmbeddingWareHouse> global_embedding_warehouse =
    std::shared_ptr<EmbeddingWareHouse>();

#endif // DAMO_EMBEDDING_EMBEDDING_H