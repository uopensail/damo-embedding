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
#define Epsilon 1e-7

#ifdef __APPLE__
#define u_int64_t __uint64_t
#elif __linux__
#define u_int64_t uint64_t
#endif

// 获得每个特征的group-id
#define groupof(x) ((x) >> 56)
#define value_mask 0xFFFFFFFFFFFFFFul
#define mask_group(group, key) ((key & value_mask) + (group << 56))

// 最大的group数量
#define max_group 256
#define min_size 2147483648
#define sign(x) ((x) >= 0.0 ? 1.0 : -1.0)
#define safe_sqrt(x) ((x) >= 0.0 ? sqrtf((x)) : 0.0)

//存放数据的结构
#pragma pack(push)
#pragma pack(1)
struct MetaData {
  u_int64_t key;
  u_int64_t update_logic_time; //更新的逻辑时间
  u_int64_t update_real_time;  //更新时间
  u_int64_t update_num;        //更新次数
  int dim;
  Float data[];
};
#pragma pack(pop)

using MetaData = struct MetaData;

static u_int64_t get_current_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return u_int64_t(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

class Params {
private:
  std::shared_ptr<cpptoml::table> table;

public:
  Params() = delete;
  Params(const std::shared_ptr<cpptoml::table> &table);
  Params(const Params &p);
  Params &operator=(const Params &p);
  //模板函数要放在头文件中，放在src中就会出现链接问题
  template <class T> T get(std::string key) const {
    if (table->contains(key)) {
      return *table->get_as<T>(key);
    }
    throw std::out_of_range(key + " is not a valid key");
  }
  ~Params();
};

#endif // DAMO_COMMON_H
