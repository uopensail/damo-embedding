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

#ifndef DAMO_EMBEDDING_COMMMON_H
#define DAMO_EMBEDDING_COMMMON_H

#pragma once

#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "json.hpp"

using json = nlohmann::json;

#define Float float
const Float Epsilon = 1e-8f;

#ifndef u_int64_t
#define u_int64_t unsigned long long
#endif

#define STEP_CONTROL_BYTESIZE 8
int64_t get_current_time();
Float safe_sqrt(Float x);
Float sign(Float x);

// the struct of value in rocksdb
#pragma pack(push)
#pragma pack(1)

struct Key {
  int group;
  int64_t key;
};

struct MetaData {
  int group;
  int64_t key;
  int64_t update_num;
  int64_t update_time; // ms
  uint64_t step_control;
  int dim;
  Float data[];
};
#pragma pack(pop)

using MetaData = struct MetaData;
using Key = struct Key;

class Params {
public:
  Params();
  Params(json &params);
  Params(const std::string &str);
  Params(const Params &p);

  Params &operator=(const Params &p);
  const bool isnil() const;
  bool contains(const std::string &key);

  template <class T> T get(const std::string &key) const {
    if (!this->params_.contains(key)) {
      throw std::out_of_range(key + " is not a valid key");
    }
    return this->params_[key].get<T>();
  }

  template <class T> void insert(const std::string &key, const T &value) const {
    this->params_[key] = value;
  }

  std::string to_json() const;

  template <class T>
  T get(const std::string &key, const T &default_value) const {
    if (!this->params_.contains(key)) {
      return default_value;
    }
    return this->params_[key].get<T>();
  }

  ~Params();

private:
  json params_;
};

#endif // DAMO_COMMON_H