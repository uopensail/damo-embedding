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
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "cpptoml.h"

#define Float float
const Float Epsilon = 1e-8f;

#ifndef uint64_t
//#define uint64_t unsigned long long
#endif

#ifndef int64_t
//#define int64_t long long
#endif

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
  int64_t update_time;  // ms
  int dim;
  Float data[];
};
#pragma pack(pop)

using MetaData = struct MetaData;
using Key = struct Key;

class Params {
 private:
  std::shared_ptr<cpptoml::table> table;

 public:
  Params() = delete;
  Params(const std::shared_ptr<cpptoml::table> &table);
  Params(const Params &p);
  const bool isnil() const;
  Params &operator=(const Params &p);
  Params &operator=(const std::shared_ptr<cpptoml::table> &table);

  bool contains(const std::string &key);

  template <class T>
  T get(const std::string &key) const {
    if (table != nullptr && table->contains(key)) {
      return *table->get_as<T>(key);
    }
    throw std::out_of_range(key + " is not a valid key");
  }

  template <class T>
  T get(const std::string &key, const T &default_value) const {
    if (table != nullptr && table->contains(key)) {
      return *table->get_as<T>(key);
    }
    return default_value;
  }
  ~Params();
};

#endif  // DAMO_COMMON_H