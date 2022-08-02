#ifndef DAMO_EMBEDDING_INITIALIZER_H
#define DAMO_EMBEDDING_INITIALIZER_H

#pragma once
#include "common.h"
#include <iostream>

class Initializer
{
protected:
    std::string name_;

public:
    Initializer() = delete;
    Initializer(const Initializer &) = delete;
    Initializer(const Initializer &&) = delete;
    Initializer(const Params &initializer_params);
    virtual ~Initializer();
    const std::string &get_name();
    virtual void call(Float *data, int &dim) = 0;
};

class Zeros : public Initializer
{
public:
    Zeros() = delete;
    Zeros(const Zeros &) = delete;
    Zeros(const Zeros &&) = delete;
    Zeros(const Params &initializer_params);
    virtual ~Zeros();
    virtual void call(Float *data, int &dim);
};

class Ones : public Initializer
{
public:
    Ones() = delete;
    Ones(const Ones &) = delete;
    Ones(const Ones &&) = delete;
    Ones(const Params &initializer_params);
    virtual ~Ones();
    virtual void call(Float *data, int &dim);
};

class RandomUniform : public Initializer
{
private:
    double min_;
    double max_;
    std::uniform_real_distribution<double> distribution;
    std::default_random_engine random;

public:
    RandomUniform() = delete;
    RandomUniform(const RandomUniform &) = delete;
    RandomUniform(const RandomUniform &&) = delete;
    RandomUniform(const Params &initializer_params);
    virtual ~RandomUniform();
    virtual void call(Float *data, int &dim);
};

class RandomNormal : public Initializer
{
private:
    double mean_;
    double stddev_;
    std::normal_distribution<double> distribution;
    std::default_random_engine random;

public:
    RandomNormal() = delete;
    RandomNormal(const RandomNormal &) = delete;
    RandomNormal(const RandomNormal &&) = delete;
    RandomNormal(const Params &initializer_params);
    virtual ~RandomNormal();
    virtual void call(Float *data, int &dim);
};

class TruncateNormal : public Initializer
{
private:
    double mean_;
    double stddev_;
    std::normal_distribution<double> distribution;
    std::default_random_engine random;

public:
    TruncateNormal() = delete;
    TruncateNormal(const TruncateNormal &) = delete;
    TruncateNormal(const TruncateNormal &&) = delete;
    TruncateNormal(const Params &initializer_params);
    virtual ~TruncateNormal();
    virtual void call(Float *data, int &dim);
};

const std::shared_ptr<Initializer> get_initializers(const Params &p);

#endif // DAMO_EMBEDDING_INITIALIZER_H
