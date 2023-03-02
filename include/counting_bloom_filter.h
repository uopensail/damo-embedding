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

#ifndef DAMO_EMBEDDING_COUNTINGBLOOMFILTER_H
#define DAMO_EMBEDDING_COUNTINGBLOOMFILTER_H

#pragma once

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <bitset>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "common.h"

const double FFP = 0.0002;
const int BitSize = 4;
const int MaxCount = (1 << BitSize) - 1;
const u_int64_t HighMask = 8589934591ull;  // 2^33-1
const u_int64_t LowMask = 2147483647ull;   // 2^31-1

using Counter = std::bitset<BitSize>;

class CountingBloomFilter : std::enable_shared_from_this<CountingBloomFilter> {
 private:
  double ffp_;            //假阳率
  size_t capacity_;       //过滤器的容量
  std::string filename_;  //持久化文件
  int count_;             //最小数量
  size_t size_;           //申请的空间大小
  int k_;                 // hash函数的个数
  int fp_;                //打开的文件描述符
  Counter *data_;         //具体的存储的数据

 public:
  CountingBloomFilter() = delete;
  CountingBloomFilter(const Params &config);
  CountingBloomFilter(const CountingBloomFilter &) = delete;
  CountingBloomFilter(size_t capacity, int count, const std::string &filename,
                      bool reload = false, double ffp = FFP);
  ~CountingBloomFilter();

 public:
  bool check(const u_int64_t &key);  //检查在不在，次数是否大于count
  void add(const u_int64_t &key, const u_int64_t &num = 1);  //添加
};
u_int64_t hash_func(const u_int64_t &x);
#endif  // DAMO_EMBEDDING_COUNTINGBLOOMFILTER_H
