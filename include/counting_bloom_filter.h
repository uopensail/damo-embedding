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

#ifndef DAMO_EMBEDDING_COUNTING_BLOOM_FILTER_H
#define DAMO_EMBEDDING_COUNTING_BLOOM_FILTER_H

#pragma once

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "common.h"

struct Counter {
  unsigned char m1 : 4;
  unsigned char m2 : 4;
};
using Counter = struct Counter;

const double FPR = 0.001;
const int max_count = 15;
const u_int64_t min_size = 268435456ull;   // 2^28
const u_int64_t high_mask = 8589934591ull; // 2^33-1
const u_int64_t low_mask = 2147483647ull;  // 2^31-1

class CountingBloomFilter final {
private:
  double fpr_;           // false positive rate
  size_t capacity_;      // The capacity of the filter
  std::string filename_; // persist files
  int count_;            // minimum count for the filter
  size_t counter_num_;   // the amount of counter
  size_t space_;         // memory space for counter
  int k_;                // the number of hash functions
  int fd_;               // file descriptor
  Counter *data_;        // stored data

public:
  CountingBloomFilter();
  CountingBloomFilter(const Params &config);
  CountingBloomFilter(const CountingBloomFilter &) = delete;
  CountingBloomFilter(size_t capacity, int count, const std::string &filename,
                      bool reload = false, double fpr = FPR);
  ~CountingBloomFilter();

public:
  /**
   * @brief where the key in the filter
   *
   * @param key key to find
   * @return true in the filter
   * @return false not in the filter
   */
  bool check(const u_int64_t &key);

  /**
   * @brief add key to the filter
   *
   * @param key key to ad
   * @param num add counts
   */
  void add(const u_int64_t &key, const u_int64_t &num = 1);
  int get_count() const;
};
u_int64_t hash_func(const u_int64_t &x);
void create_empty_file(const std::string &filename, const size_t &size);

#endif // DAMO_EMBEDDING_COUNTING_BLOOM_FILTER_H