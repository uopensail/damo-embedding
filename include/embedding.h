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

#ifndef DAMO_EMBEDDING_EMBEDDING_H_
#define DAMO_EMBEDDING_EMBEDDING_H_

#pragma once

#include <rocksdb/db.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/utilities/db_ttl.h>
#include <rocksdb/write_batch.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "initializer.h"
#include "optimizer.h"

namespace embedding {

constexpr int kMaxEmbeddingNum = 256;  ///< Maximum supported embeddings

/**
 * @brief Represents a single embedding configuration.
 */
class Embedding {
 public:
  Embedding() = delete;
  explicit Embedding(const nlohmann::json& configure);
  ~Embedding() = default;

  // Accessors
  int dimension() const noexcept { return dim_; }
  int group() const noexcept { return group_; }
  std::shared_ptr<Optimizer> optimizer() const noexcept { return optimizer_; }
  std::shared_ptr<Initializer> initializer() const noexcept {
    return initializer_;
  }

 private:
  int dim_;                               ///< Dimension of the embedding
  int group_;                             ///< Group identifier
  std::shared_ptr<Optimizer> optimizer_;  ///< Optimization algorithm
  std::shared_ptr<Initializer>
      initializer_;  ///< Weight initialization strategy
};

/**
 * @brief RocksDB merge operator for gradient application.
 */
class ApplyGradientsOperator : public rocksdb::MergeOperator {
 public:
  explicit ApplyGradientsOperator(const nlohmann::json& configure);
  ~ApplyGradientsOperator() override = default;

  bool FullMerge(const rocksdb::Slice& key,
                 const rocksdb::Slice* existing_value,
                 const std::deque<std::string>& operand_list,
                 std::string* new_value,
                 rocksdb::Logger* logger) const override;

  bool PartialMerge(const rocksdb::Slice& key,
                    const rocksdb::Slice& left_operand,
                    const rocksdb::Slice& right_operand, std::string* new_value,
                    rocksdb::Logger* logger) const override {
    return false;
  }

  const char* Name() const override { return "ApplyGradientsOperator"; }

 private:
  std::array<std::unique_ptr<Embedding>, kMaxEmbeddingNum> embeddings_;
};

/**
 * @brief Manages embedding storage and operations.
 */
class EmbeddingWarehouse {
 public:
  EmbeddingWarehouse() = delete;
  explicit EmbeddingWarehouse(const nlohmann::json& configure);
  ~EmbeddingWarehouse() = default;

  std::string to_json() const { return this->configure_.dump(); }
  void dump(const std::string& path);
  void checkpoint(const std::string& path);
  void load(const std::string& path);

  void lookup(int group, const int64_t* keys, int len, float* data,
              int n) const;

  void apply_gradients(int group, const int64_t* keys, int len,
                       const float* grads, int n);

  int dimension(int group) const;

 private:
  // RAII封装的内存映射指针
  struct RocksdbDeleter {
    void operator()(rocksdb::DBWithTTL* db) const;
  };
  using RocksdbPtr = std::unique_ptr<rocksdb::DBWithTTL, RocksdbDeleter>;

  std::unique_ptr<std::string> create_record(int group, int64_t key) const;

  nlohmann::json configure_;  ///< Configuration parameters
  int size_;                  ///< Number of embeddings
  std::array<std::unique_ptr<Embedding>, kMaxEmbeddingNum>
      embeddings_;  ///< Embedding configurations

  RocksdbPtr db_;             ///< RocksDB instance with TTL support
  mutable std::mutex mutex_;  ///< Synchronization primitive
};

}  // namespace embedding

#endif  // DAMO_EMBEDDING_EMBEDDING_H_