#include "optimizer.h"

Optimizer::Optimizer(const Params &optimizer_params, const Params &decay_params) : name_(optimizer_params.get<std::string>("name")),
                                                                                   function_(get_decay_lr_func(decay_params)),
                                                                                   decay_params_(decay_params) {}

const std::string &Optimizer::get_name() { return name_; }

Optimizer::~Optimizer(){};

void Optimizer::init_helper(Float *data, int dim) {}

int Optimizer::get_space(int dim) { return dim; }

inline Float Optimizer::get_decay_rate(u_int64_t global_step, Float learning_rate_)
{
    if (function_)
    {
        return function_(learning_rate_, global_step, decay_params_);
    }
    return 0.0;
}

SGDOptimizer::SGDOptimizer(const Params &optimizer_params, const Params &decay_params) : Optimizer(optimizer_params, decay_params),
                                                                                         eta(optimizer_params.get<double>("eta")) {}

SGDOptimizer::~SGDOptimizer() {}

void SGDOptimizer::call(Float *data, Float *gds, int dim, u_int64_t global_step)
{
    auto decay = get_decay_rate(global_step, eta);
    for (int i = 0; i < dim; i++)
    {
        data[i] -= eta * gds[i] + decay * data[i];
    }
}

FTRLOptimizer::FTRLOptimizer(const Params &optimizer_params, const Params &decay_params) : Optimizer(optimizer_params, decay_params),
                                                                                           alpha(optimizer_params.get<double>("alpha")),
                                                                                           beta(optimizer_params.get<double>("beta")),
                                                                                           lambda1(optimizer_params.get<double>("lambda1")),
                                                                                           lambda2(optimizer_params.get<double>("lambda2"))
{
}

FTRLOptimizer::~FTRLOptimizer() {}

int FTRLOptimizer::get_space(int dim) { return 3 * dim; }

void FTRLOptimizer::call(Float *data, Float *gds, int dim, u_int64_t global_step)
{
    Float *w = data;
    Float *z = &(data[dim]);
    Float *n = &(data[dim << 1]);
    Float tmp1, tmp2, delta, eta;
    for (int i = 0; i < dim; i++)
    {
        tmp1 = n[i] + gds[i] * gds[i];
        delta = ((safe_sqrt(tmp1) - safe_sqrt(n[i]))) / alpha;
        z[i] += gds[i] - delta * w[i];
        tmp2 = abs(z[i]);
        if (tmp2 <= lambda1)
        {
            w[i] = 0;
        }
        else
        {
            n[i] = tmp1;
            eta = -1.0 / ((beta + safe_sqrt(n[i])) / alpha + lambda2);
            w[i] = eta * (z[i] - sign(z[i]) * lambda1);
        }
    }
}

AdamOptimizer::AdamOptimizer(const Params &optimizer_params, const Params &decay_params) : Optimizer(optimizer_params, decay_params),
                                                                                           alpha(optimizer_params.get<double>("alpha")),
                                                                                           beta1(optimizer_params.get<double>("beta1")),
                                                                                           beta2(optimizer_params.get<double>("beta2")),
                                                                                           epsilon(Epsilon)
{
    try
    {
        epsilon = optimizer_params.get<double>("epsilon");
    }
    catch (...)
    {
    }
}

AdamOptimizer::~AdamOptimizer() {}

int AdamOptimizer::get_space(int dim) { return 3 * dim + 2; }

void AdamOptimizer::init_helper(Float *data, int dim)
{
    data[dim * 3] = 1.0;
    data[dim * 3 + 1] = 1.0;
}

void AdamOptimizer::call(Float *data, Float *gds, int dim, u_int64_t global_step)
{
    Float *w = data;
    Float *m = &(data[dim]);
    Float *v = &(data[dim << 1]);
    auto decay = get_decay_rate(global_step, alpha);
    Float &beta1_t = data[dim * 3], &beta2_t = data[dim * 3 + 1];
    beta1_t = beta1_t <= epsilon ? 0.0 : beta1_t * beta1;
    beta2_t = beta2_t <= epsilon ? 0.0 : beta2_t * beta2;
    Float lr = alpha * safe_sqrt(1.0 - beta2_t) / (1.0 - beta1_t);

    for (int i = 0; i < dim; i++)
    {
        m[i] = beta1 * m[i] + (1.0 - beta1) * gds[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * gds[i] * gds[i];

        w[i] -= lr * m[i] / (safe_sqrt(v[i]) + epsilon) + decay * w[i];
    }
}

AmsGradOptimizer::AmsGradOptimizer(const Params &optimizer_params, const Params &decay_params) : Optimizer(optimizer_params, decay_params),
                                                                                                 alpha(optimizer_params.get<double>("alpha")),
                                                                                                 beta1(optimizer_params.get<double>("beta1")),
                                                                                                 beta2(optimizer_params.get<double>("beta2")),
                                                                                                 epsilon(Epsilon)
{
    try
    {
        epsilon = optimizer_params.get<double>("epsilon");
    }
    catch (...)
    {
    }
}

AmsGradOptimizer::~AmsGradOptimizer() {}

int AmsGradOptimizer::get_space(int dim) { return 4 * dim + 2; }

void AmsGradOptimizer::init_helper(Float *data, int dim)
{
    data[dim * 3] = 1.0;
    data[dim * 3 + 1] = 1.0;
}

void AmsGradOptimizer::call(Float *data, Float *gds, int dim, u_int64_t global_step)
{
    Float *w = data;
    Float *m = &(data[dim]);
    Float *v = &(data[dim * 2]);
    Float *max_v = &(data[dim * 3]);

    auto decay = get_decay_rate(global_step, alpha);
    Float &beta1_t = data[dim * 3], &beta2_t = data[dim * 3 + 1];
    beta1_t = beta1_t <= epsilon ? 0.0 : beta1_t * beta1;
    beta2_t = beta2_t <= epsilon ? 0.0 : beta2_t * beta2;
    Float lr = alpha * safe_sqrt(1.0 - beta2_t) / (1.0 - beta1_t);
    for (int i = 0; i < dim; i++)
    {
        m[i] = beta1 * m[i] + (1.0 - beta1) * gds[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * gds[i] * gds[i];
        max_v[i] = max_v[i] < v[i] ? v[i] : max_v[i];
        w[i] -= lr * m[i] / (safe_sqrt(max_v[i]) + epsilon) + decay * w[i];
    }
}

const std::shared_ptr<Optimizer> get_optimizers(const Params &optimizer_params, const Params &decay_params)
{
    auto name = optimizer_params.get<std::string>("name");
    if (name == "sgd")
    {
        return std::shared_ptr<Optimizer>(new SGDOptimizer{optimizer_params, decay_params});
    }
    else if (name == "ftrl")
    {
        return std::shared_ptr<Optimizer>(new FTRLOptimizer{optimizer_params, decay_params});
    }
    else if (name == "adam")
    {
        return std::shared_ptr<Optimizer>(new AdamOptimizer{optimizer_params, decay_params});
    }
    else if (name == "amsgrad")
    {
        return std::shared_ptr<Optimizer>(new AmsGradOptimizer{optimizer_params, decay_params});
    }
    else
    {
        std::cout << "No Such Optimizer: " << name << std::endl;
        exit(-3);
    }
}
