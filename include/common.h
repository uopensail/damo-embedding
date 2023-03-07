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

using u_int64_t = unsigned long long;

// the feature number must <= 256
const int max_group = 256;
const u_int64_t min_size = 2147483648ull;

u_int64_t get_current_time();
Float safe_sqrt(Float x);
Float sign(Float x);

// get the group id from the key
inline u_int64_t groupof(const u_int64_t &key);

// the struct of value in rocksdb
#pragma pack(push)
#pragma pack(1)
struct MetaData {
  u_int64_t key;
  u_int64_t update_time;  // ms
  u_int64_t update_num;
  int dim;
  Float data[];
};
#pragma pack(pop)

using MetaData = struct MetaData;

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

  template <class T>
  T get(const std::string &key) const {
    if (table != null && table->contains(key)) {
      return *table->get_as<T>(key);
    }
    throw std::out_of_range(key + " is not a valid key");
  }

  template <class T>
  T get(const std::string &key, const T &default_value) const {
    if (table != null && table->contains(key)) {
      return *table->get_as<T>(key);
    }
    return default_value;
  }
  ~Params();
};

#endif  // DAMO_COMMON_H
