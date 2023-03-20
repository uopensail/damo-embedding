#include "pyembedding.h"

Parameters::Parameters() : params_(cpptoml::make_table()) {}
Parameters::Parameters(const Parameters &p) : params_(p.params_){};
Parameters &Parameters::operator=(const Parameters &p) {
  if (this == &p) {
    return *this;
  }
  this->params_ = p.params_;
  return *this;
}

Parameters ::~Parameters() {}

void Parameters::insert(std::string key, bool value) {
  auto value_ = cpptoml::make_value<bool>(std::move(value));
  this->params_->insert(key, value_);
}

void Parameters::insert(std::string key, int value) {
  int64_t tmp = value;
  auto value_ = cpptoml::make_value<int64_t>(std::move(tmp));
  this->params_->insert(key, value_);
}

void Parameters::insert(std::string key, std::string value) {
  auto value_ = cpptoml::make_value<std::string>(std::move(value));
  this->params_->insert(key, value_);
}

void Parameters::insert(std::string key, double value) {
  auto value_ = cpptoml::make_value<double>(std::move(value));
  this->params_->insert(key, value_);
}

// PyInitializer的实现
PyInitializer::PyInitializer() {
  Parameters p;
  p.insert("name", std::string("zeros"));
  this->initializer_ = get_initializers(Params(p.params_));
}

PyInitializer::PyInitializer(Parameters params) {
  this->initializer_ =
      std::shared_ptr<Initializer>(get_initializers(Params(params.params_)));
}

PyInitializer::PyInitializer(const PyInitializer &p)
    : initializer_(p.initializer_) {}

PyInitializer &PyInitializer::operator=(const PyInitializer &p) {
  if (this == &p) {
    return *this;
  }
  this->initializer_ = p.initializer_;
  return *this;
}

void PyInitializer::call(float *w, int wn) { initializer_->call(w, wn); }

PyInitializer::~PyInitializer() {}

// PyOptimizer的实现
PyOptimizer::PyOptimizer() {
  Parameters sgd, decay;
  sgd.insert("name", std::string("sgd"));
  decay.insert("name", std::string(""));
  this->optimizer_ = get_optimizers(Params(sgd.params_), Params(decay.params_));
}

PyOptimizer::PyOptimizer(Parameters op_params) {
  Parameters decay;
  decay.insert("name", std::string(""));
  optimizer_ = get_optimizers(Params(op_params.params_), Params(decay.params_));
}

PyOptimizer::PyOptimizer(Parameters op_params, Parameters decay_params) {
  optimizer_ =
      get_optimizers(Params(op_params.params_), Params(decay_params.params_));
}

PyOptimizer::PyOptimizer(const PyOptimizer &p) : optimizer_(p.optimizer_) {}

PyOptimizer &PyOptimizer::operator=(const PyOptimizer &p) {
  if (this == &p) {
    return *this;
  }
  this->optimizer_ = p.optimizer_;
  return *this;
}
PyOptimizer::~PyOptimizer() {}

void PyOptimizer::call(float *data, int wn, float *gds, int gn,
                       unsigned long long global_step) {
  assert(optimizer_->get_space(gn) == wn);
  optimizer_->call(data, gds, wn, global_step);
}

// PyFilter的实现
PyFilter::PyFilter() : filter_(nullptr) {}
PyFilter::PyFilter(Parameters params)
    : filter_(std::make_shared<CountingBloomFilter>(params.params_)) {}

PyFilter::PyFilter(const PyFilter &p) : filter_(p.filter_) {}

PyFilter &PyFilter::operator=(const PyFilter &p) {
  if (this == &p) {
    return *this;
  }
  this->filter_ = p.filter_;
  return *this;
}

bool PyFilter::check(unsigned long long key) {
  if (this->filter_ == nullptr) {
    return true;
  }
  return filter_->check(key);
}

void PyFilter::add(unsigned long long key, unsigned long long num) {
  if (this->filter_ != nullptr) {
    this->filter_->add(key, num);
  }
}

PyFilter::~PyFilter() {}

PyStorage::PyStorage(const std::string &data_dir, int ttl = 0) {
  this->storage_ = std::make_shared<Storage>(ttl, data_dir);
}

PyStorage::~PyStorage() {}

void PyStorage::dump(const std::string &path, int expires, int group) {
  auto func = [&group, &expires](MetaData *ptr) -> bool {
    if (ptr == nullptr) {
      return false;
    }
    auto oldest_ts = get_current_time() - expires * 86400;
    if (ptr->update_time < oldest_ts) {
      return false;
    }
    if ((group == -1) ||
        (0 <= group && group < max_group && group == groupof(ptr->key))) {
      return true;
    }
    return false;
  };
  this->storage_->dump(path, func);
}

PyEmbedding::PyEmbedding(PyStorage storage, PyOptimizer optimizer,
                         PyInitializer initializer, int dim, int count) {
  this->embedding_ =
      std::make_unique<Embedding>(storage.storage_, optimizer.optimizer_,
                                  initializer.initializer_, dim, count);
}

PyEmbedding::PyEmbedding(const PyEmbedding &p) {
  if (&p != this) {
    this->embedding_ = p.embedding_;
  }
}

PyEmbedding &PyEmbedding::operator=(const PyEmbedding &p) {
  if (&p != this) {
    this->embedding_ = p.embedding_;
  }
  return *this;
}

PyEmbedding::~PyEmbedding() {}

void PyEmbedding::lookup(unsigned long long *keys, int len, float *data,
                         int n) {
  this->embedding_->lookup((u_int64_t *)keys, len, data, n);
}

void PyEmbedding::apply_gradients(unsigned long long *keys, int len, float *gds,
                                  int n, unsigned long long global_step) {
  this->embedding_->apply_gradients((u_int64_t *)keys, len, gds, n,
                                    global_step);
}
