#ifndef DAMO_EMBEDDING_PY_EMBEDDING_H
#define DAMO_EMBEDDING_PY_EMBEDDING_H

#pragma once

#include "count_bloom_filter.h"
#include "cpptoml.h"
#include "embedding.h"
#include "initializer.h"
#include "optimizer.h"

class Parameters
{
public:
    std::shared_ptr<cpptoml::table> params_;

public:
    Parameters();
    Parameters(const Parameters &p);
    Parameters(const Parameters &&p);
    Parameters &operator=(const Parameters &p);
    ~Parameters();
    void insert(std::string key, std::string value);
    void insert(std::string key, int value);
    void insert(std::string key, double value);
};

class PyEmbedding;
class PyInitializer
{
private:
    std::shared_ptr<Initializer> initializer_;
    friend class PyEmbedding;

public:
    PyInitializer();
    PyInitializer(Parameters params);
    PyInitializer(const PyInitializer &p);
    PyInitializer(const PyInitializer &&p);
    PyInitializer &operator=(const PyInitializer &p);
    ~PyInitializer();
};

class PyOptimizer
{
private:
    std::shared_ptr<Optimizer> optimizer_;
    friend class PyEmbedding;

public:
    PyOptimizer();
    PyOptimizer(Parameters op_params);
    PyOptimizer(Parameters op_params, Parameters decay_params);
    PyOptimizer(const PyOptimizer &p);
    PyOptimizer(const PyOptimizer &&p);
    PyOptimizer &operator=(const PyOptimizer &p);
    ~PyOptimizer();
};

class PyFilter
{
private:
    std::shared_ptr<CountBloomFilter> filter_;
    friend class PyEmbedding;

public:
    PyFilter();
    PyFilter(Parameters params);
    PyFilter(const PyFilter &p);
    PyFilter(const PyFilter &&p);
    PyFilter &operator=(const PyFilter &p);
    ~PyFilter();
};

class PyEmbedding
{
private:
    std::shared_ptr<Embedding> embedding_;

public:
    PyEmbedding() = delete;
    PyEmbedding(const PyEmbedding &p) = delete;
    PyEmbedding(const PyEmbedding &&p) = delete;
    PyFilter &operator=(const PyFilter &p) = delete;

public:
    // 初始化embedding
    // dim: 维度
    // data_dir: 数据存放的路径
    // filter: 频控, 可为空
    // optimizer: 优化算子
    // initializer: 初始化算子
    PyEmbedding(int dim, std::string data_dir, PyFilter filter,
                PyOptimizer optimizer, PyInitializer initializer);
    ~PyEmbedding();
    // 查询
    // keys: 需要查询的keys
    // len: keys的长度
    // data: 返回的数据
    // n: 返回的数据长度
    // global_step: 全局step
    void lookup(u_int64_t *keys, int len, Float *data, int n, u_int64_t &global_step);
    void apply_gradients(u_int64_t *keys, int len, Float *gds, int n, u_int64_t global_step);
    void dump(std::string path, int expires);
};

#endif // DAMO_EMBEDDING_PY_EMBEDDING_H