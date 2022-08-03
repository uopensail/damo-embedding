#ifndef DAMO_EMBEDDING_COMMMON_H
#define DAMO_EMBEDDING_COMMMON_H

#pragma once

#include "cpptoml.h"
#include <algorithm>
#include <math.h>
#include <memory>
#include <random>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <unordered_map>

#define Float float
#define Epsilon 1e-7

#ifdef __APPLE__
#define u_int64_t __uint64_t
#elif __linux__
#define u_int64_t uint64_t
#endif

// 获得每个特征的group-id
#define groupof(x) ((x) >> 56)

// 最大的group数量
#define max_group 256
#define min_size 2147483648
#define sign(x) ((x) >= 0.0 ? 1.0 : -1.0)
#define safe_sqrt(x) ((x) >= 0.0 ? sqrt((x)) : 0.0)

//存放数据的结构
#pragma pack(push)
#pragma pack(1)
struct MetaData
{
    u_int64_t key;
    u_int64_t update_logic_time; //更新的逻辑时间
    u_int64_t update_real_time;  //更新时间
    u_int64_t update_num;        //更新次数
    int dim;
    Float data[];
};
#pragma pack(pop)

using MetaData = struct MetaData;

class Params
{
private:
    std::shared_ptr<cpptoml::table> table;

public:
    Params()=delete;
    Params(std::shared_ptr<cpptoml::table> &table);
    Params(std::shared_ptr<cpptoml::table> &&table);
    Params(const Params &p);
    Params(const Params &&p);
    Params &operator=(const Params &p);
    //模板函数要放在头文件中，放在src中就会出现链接问题
    template <class T>
    T get(std::string key) const
    {
        if (table->contains(key))
        {
            return *table->get_as<T>(key);
        }
        throw std::out_of_range(key + " is not a valid key");
    }
    ~Params();
};

u_int64_t get_current_time();

#endif // DAMO_COMMON_H
