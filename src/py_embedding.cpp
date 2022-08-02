#include "py_embedding.h"

Parameters::Parameters() : params_(cpptoml::make_table()) {}
Parameters::Parameters(const Parameters &p) : params_(p.params_){};
Parameters::Parameters(const Parameters &&p) : params_(p.params_){};
Parameters &Parameters::operator=(const Parameters &p)
{
    this->params_ = p.params_;
    return *this;
}

Parameters ::~Parameters()
{
}

void Parameters::insert(std::string key, int value)
{
    int64_t tmp = value;
    auto value_ = cpptoml::make_value<int>(std::move(tmp));
    params_->insert(key, value_);
}

void Parameters::insert(std::string key, std::string value)
{
    std::shared_ptr<cpptoml::value<std::string>> value_ = cpptoml::make_value<std::string>(std::move(value));
    params_->insert(key, value_);
}

void Parameters::insert(std::string key, double value)
{
    std::shared_ptr<cpptoml::value<double>> value_ = cpptoml::make_value<double>(std::move(value));
    params_->insert(key, value_);
}

// PyInitializer的实现
PyInitializer::PyInitializer()
{
    Parameters p;
    p.insert("name", "zeros");
    initializer_ = get_initializers(Params(p.params_));
}

PyInitializer::PyInitializer(Parameters params)
{
    initializer_ = std::shared_ptr<Initializer>(get_initializers(Params(params.params_)));
}

PyInitializer::PyInitializer(const PyInitializer &p) : initializer_(p.initializer_) {}

PyInitializer::PyInitializer(const PyInitializer &&p) : initializer_(p.initializer_) {}

PyInitializer &PyInitializer::operator=(const PyInitializer &p)
{
    this->initializer_ = p.initializer_;
    return *this;
}

PyInitializer::~PyInitializer() {}

// PyOptimizer的实现
PyOptimizer::PyOptimizer()
{
    Parameters sgd, decay;
    sgd.insert("name", "sgd");
    sgd.insert("eta", 0.001);
    decay.insert("name", "");
    optimizer_ = get_optimizers(Params(sgd.params_), Params(decay.params_));
}

PyOptimizer::PyOptimizer(Parameters op_params)
{
    Parameters decay;
    decay.insert("name", "");
    optimizer_ = get_optimizers(Params(op_params.params_), Params(decay.params_));
}

PyOptimizer::PyOptimizer(Parameters op_params, Parameters decay_params)
{
    optimizer_ = get_optimizers(Params(op_params.params_), Params(decay_params.params_));
}

PyOptimizer::PyOptimizer(const PyOptimizer &p) : optimizer_(p.optimizer_) {}
PyOptimizer::PyOptimizer(const PyOptimizer &&p) : optimizer_(p.optimizer_) {}
PyOptimizer &PyOptimizer::operator=(const PyOptimizer &p)
{
    this->optimizer_ = p.optimizer_;
    return *this;
}
PyOptimizer::~PyOptimizer() {}

// PyFilter的实现
PyFilter::PyFilter() : filter_(nullptr) {}
PyFilter::PyFilter(Parameters params)
{
    size_t capacity = params.params_->get_as<size_t>("capacity").value_or(min_size);
    int count = params.params_->get_as<int>("count").value_or(1);
    std::string filename = params.params_->get_as<std::string>("filename").value_or("/tmp/COUNTBLOOMFILTERDATA");
    bool reload = params.params_->get_as<bool>("reload").value_or(true);
    double ffp = params.params_->get_as<double>("ffp").value_or(FFP);
    filter_ = std::shared_ptr<CountBloomFilter>(new CountBloomFilter(capacity, count, filename, reload, ffp));
}

PyFilter::PyFilter(const PyFilter &p) : filter_(p.filter_) {}
PyFilter::PyFilter(const PyFilter &&p) : filter_(p.filter_) {}
PyFilter &PyFilter::operator=(const PyFilter &p)
{
    this->filter_ = p.filter_;
    return *this;
}
PyFilter::~PyFilter() {}

// PyEmbedding实现
PyEmbedding::PyEmbedding(int dim, std::string data_dir, PyFilter filter,
                         PyOptimizer optimizer, PyInitializer initializer)
{
}
PyEmbedding::~PyEmbedding() {}

void PyEmbedding::lookup(u_int64_t *keys, int len, Float *data, int n, u_int64_t &global_step)
{
    embedding_->lookup(keys, len, data, n, global_step);
}
void PyEmbedding::apply_gradients(u_int64_t *keys, int len, Float *gds, int n, u_int64_t global_step)
{
    embedding_->apply_gradients(keys, len, gds, n, global_step);
}
void PyEmbedding::dump(std::string path, int expires)
{
    embedding_->dump(path, expires);
}