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

#ifndef DAMO_EMBEDDING_COUNTBLOOMFILTER_H
#define DAMO_EMBEDDING_COUNTBLOOMFILTER_H

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
#include <thread>

#include "common.h"

#define FFP 0.0002
#define BitSize 4
#define MaxCount (1 << BitSize) - 1
#define HighMask 8589934591ull  // 2^33-1
#define LowMask 2147483647ull   // 2^31-1
#define hash_func(x)                                \
  (((static_cast<u_int64_t>(x) >> 31) & HighMask) | \
   (static_cast<u_int64_t>(x) & LowMask) << 33)

using Counter = std::bitset<BitSize>;

//定义全局的线程状态
static std::atomic<bool> CountBloomFilterGlobalStatus(true);

class CountBloomFilter {
 private:
  double ffp_;                               //假阳率
  size_t capacity_;                          //过滤器的容量
  std::string filename_;                     //持久化文件
  int count_;                                //最小数量
  size_t size_;                              //申请的空间大小
  int k_;                                    // hash函数的个数
  int fp_;                                   //打开的文件描述符
  Counter *data_;                            //具体的存储的数据
  std::thread flush_thread_;                 //定期刷到磁盘的线程
  std::thread::native_handle_type handler_;  //退出线程的处理

 public:
  CountBloomFilter() = delete;
  CountBloomFilter(const Params &config);
  CountBloomFilter(const CountBloomFilter &) = delete;
  CountBloomFilter(size_t capacity, int count, std::string filename,
                   bool reload = false, double ffp = FFP);
  ~CountBloomFilter();

 public:
  void dump();                       // mmp的数据写入磁盘
  bool check(const u_int64_t &key);  //检查在不在，次数是否大于count
  void add(const u_int64_t &key, u_int64_t num = 1);  //添加
};

void flush_thread_func(CountBloomFilter *filter);

#endif  // DAMO_EMBEDDING_COUNTBLOOMFILTER_H
