#ifndef DAMO_EMBEDDING_COUNTBLOOMFILTER_H
#define DAMO_EMBEDDING_COUNTBLOOMFILTER_H

#pragma once

#include "common.h"
#include <chrono>
#include <atomic>
#include <cstdio>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

#define FFP 0.0002
#define MaxCount 15
#define HighMask 8589934591ul // 2^33-1
#define LowMask 2147483647ul  // 2^31-1
#define hash_func(x) ((((x) >> 31) & HighMask) + ((x)&LowMask) << 33)

struct BiCounter
{
    unsigned char m1 : 4;
    unsigned char m2 : 4;
};

using BiCounter = struct BiCounter;

//定义全局的线程状态
static std::atomic<bool> CountBloomFilterGlobalStatus(true);

class CountBloomFilter
{
private:
    double ffp_; //假阳率
    int k;       // hash函数的个数
    size_t size_;
    size_t capacity_;
    BiCounter *data_;
    std::string filename_; //持久化文件
    int count_;            //最小数量
    int fp_;               //打开的文件描述符
    std::thread flush_thread_;
    std::thread::native_handle_type handler_;

public:
    CountBloomFilter() = delete;
    CountBloomFilter(const CountBloomFilter &) = delete;
    CountBloomFilter(const CountBloomFilter &&) = delete;
    CountBloomFilter(size_t capacity, int count, std::string filename, bool reload = false, double ffp = FFP);
    void dump();
    //检查在不在，次数是否大于count
    bool check(const u_int64_t &key);
    void add(const u_int64_t &key, u_int64_t num = 1);
    ~CountBloomFilter();
};

void flush_thread_func(CountBloomFilter *filter);

#endif // DAMO_EMBEDDING_COUNTBLOOMFILTER_H
