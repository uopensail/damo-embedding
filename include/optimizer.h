#ifndef DAMO_EMBEDDING_OPTIMIZER_H
#define DAMO_EMBEDDING_OPTIMIZER_H

#pragma once

#include "common.h"
#include "decay_learning_rate.h"
#include "initializer.h"
#include <iostream>

class Optimizer
{
protected:
    std::string name_;
    decay_lr_func function_;
    Params decay_params_;

public:
    Optimizer() = delete;
    Optimizer(const Optimizer &) = delete;
    Optimizer(const Optimizer &&) = delete;
    Optimizer(const Params &optimizer_params, const Params &decay_params);
    virtual ~Optimizer();
    const std::string &get_name();
    virtual int get_space(int dim);
    //初始化相关操作
    virtual void init_helper(Float *data, int dim);
    inline Float get_decay_rate(u_int64_t global_step, Float learning_rate);
    virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step) = 0;
};

class SGDOptimizer : public Optimizer
{
private:
    Float eta;

public:
    SGDOptimizer() = delete;
    SGDOptimizer(const SGDOptimizer &) = delete;
    SGDOptimizer(const SGDOptimizer &&) = delete;
    SGDOptimizer(const Params &optimizer_params, const Params &decay_params);
    virtual ~SGDOptimizer();
    virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

class FTRLOptimizer : public Optimizer
{
private:
    Float alpha;
    Float beta;
    Float lambda1;
    Float lambda2;

public:
    FTRLOptimizer() = delete;
    FTRLOptimizer(const FTRLOptimizer &) = delete;
    FTRLOptimizer(const FTRLOptimizer &&) = delete;
    FTRLOptimizer(const Params &optimizer_params, const Params &decay_params);
    virtual ~FTRLOptimizer();
    virtual int get_space(int dim);
    virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

class AdamOptimizer : public Optimizer
{
private:
    Float alpha;
    Float beta1;
    Float beta2;
    Float epsilon;

public:
    AdamOptimizer() = delete;
    AdamOptimizer(const AdamOptimizer &) = delete;
    AdamOptimizer(const AdamOptimizer &&) = delete;
    AdamOptimizer(const Params &optimizer_params, const Params &decay_params);
    virtual ~AdamOptimizer();
    virtual int get_space(int dim);
    virtual void init_helper(Float *data, int dim);
    virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

class AmsGradOptimizer : public Optimizer
{
private:
    Float alpha;
    Float beta1;
    Float beta2;
    Float epsilon;

public:
    AmsGradOptimizer() = delete;
    AmsGradOptimizer(const AmsGradOptimizer &) = delete;
    AmsGradOptimizer(const AmsGradOptimizer &&) = delete;
    AmsGradOptimizer(const Params &optimizer_params, const Params &decay_params);
    virtual ~AmsGradOptimizer();
    virtual int get_space(int dim);
    virtual void init_helper(Float *data, int dim);
    virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

const std::shared_ptr<Optimizer> get_optimizers(const Params &optimizer_params, const Params &decay_params);

#endif // DAMO_EMBEDDING_OPTIMIZER_H
