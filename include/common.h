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

#ifndef DAMO_EMBEDDING_COMMMON_H_
#define DAMO_EMBEDDING_COMMMON_H_

#pragma once

#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "json.hpp"

namespace embedding {

constexpr float kEpsilon = 1e-8f;

/**
 * @brief Gets the current timestamp in milliseconds.
 * @return The current time in milliseconds.
 */
int64_t get_current_time();

/**
 * @brief Safely computes the square root.
 * @param x The input value.
 * @return The result of the square root.
 * @exception std::invalid_argument Thrown when the input is negative.
 */
float safe_sqrt(float x);

/**
 * @brief Returns the sign of a number.
 * @param x The input value.
 * @return 1 for positive, -1 for negative, 0 for zero.
 */
float sign(float x);

#pragma pack(push, 1)

/// @brief Structure definition for a key.
struct Key {
  int group;    ///< Group identifier
  int64_t key;  ///< Key value
};

/// @brief Structure definition for metadata.
struct MetaData {
  int32_t group;        ///< Group identifier
  int64_t key;          ///< Key value
  int64_t update_num;   ///< Number of updates
  int64_t update_time;  ///< Last update time in milliseconds
  int32_t dim;          ///< Data dimension
  float data[];         ///< Flexible array to store data
};
#pragma pack(pop)

/// @brief Class for handling parameters.
class Params {
 public:
  Params() = default;

  /// @brief Constructs from a JSON object.
  explicit Params(const nlohmann::json& params);

  /// @brief Constructs from a JSON string.
  explicit Params(const std::string& str);

  // Delete copy constructor and copy assignment operator
  Params(const Params&) = delete;
  Params& operator=(const Params&) = delete;

  // Allow move semantics
  Params(Params&&) = default;
  Params& operator=(Params&&) = default;

  ~Params() = default;

  /// @brief Checks if a key exists in the parameters.
  bool contains(const std::string& key) const { return params_.contains(key); }

  /// @brief Gets the value of a specified key.
  /// @tparam T The return value type.
  /// @param key The key to query.
  /// @return The corresponding value.
  /// @throws std::out_of_range Thrown when the key does not exist.
  template <class T>
  T get(const std::string& key) const {
    if (!params_.contains(key)) {
      throw std::out_of_range(key + " not found in parameters");
    }
    return params_[key].get<T>();
  }

  /// @brief Gets the value of a specified key with a default.
  template <class T>
  T get(const std::string& key, const T& default_value) const {
    if (!params_.contains(key)) {
      return default_value;
    }
    return params_[key].get<T>();
  }

  /// @brief Inserts or updates a parameter.
  template <class T>
  void insert(const std::string& key, T&& value) {
    params_[key] = std::forward<T>(value);
  }

  /// @brief Converts parameters to a JSON string.
  std::string to_json() const { return params_.dump(); }

 private:
  nlohmann::json params_;
};

}  // namespace embedding

#endif  // DAMO_EMBEDDING_COMMMON_H_