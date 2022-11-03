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

void PyInitializer::call(Float *data, int dim) {
  initializer_->call(data, dim);
}

PyInitializer::~PyInitializer() {}

// PyOptimizer的实现
PyOptimizer::PyOptimizer() {
  Parameters sgd, decay;
  sgd.insert("name", std::string("sgd"));
  sgd.insert("eta", double(0.001));
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
void PyOptimizer::call(Float *data, int wn, Float *gds, int gn,
                       u_int64_t global_step) {
  assert(wn == gn);
  optimizer_->call(data, gds, wn, global_step);
}
// PyFilter的实现
PyFilter::PyFilter() : filter_(nullptr) {}
PyFilter::PyFilter(Parameters params) {
  size_t capacity =
      params.params_->get_as<size_t>("capacity").value_or(min_size);
  int count = params.params_->get_as<int>("count").value_or(1);
  std::string filename = params.params_->get_as<std::string>("filename")
                             .value_or("/tmp/COUNTBLOOMFILTERDATA");
  bool reload = params.params_->get_as<bool>("reload").value_or(true);
  double ffp = params.params_->get_as<double>("ffp").value_or(FFP);
  filter_ = std::shared_ptr<CountBloomFilter>(
      new CountBloomFilter(capacity, count, filename, reload, ffp));
}

PyFilter::PyFilter(const PyFilter &p) : filter_(p.filter_) {}

PyFilter &PyFilter::operator=(const PyFilter &p) {
  if (this == &p) {
    return *this;
  }
  this->filter_ = p.filter_;
  return *this;
}

bool PyFilter::check(u_int64_t key) {
  if (filter_ == nullptr) {
    return true;
  }
  return filter_->check(key);
}
void PyFilter::add(u_int64_t key, u_int64_t num) {
  if (filter_ != nullptr) {
    filter_->add(key, num);
  }
}

PyFilter::~PyFilter() {}

// PyEmbedding实现
PyEmbedding::PyEmbedding(int dim, unsigned long long max_lag,
                         std::string data_dir, PyFilter filter,
                         PyOptimizer optimizer, PyInitializer initializer) {
  auto empedding_ptr =
      new Embedding(dim, (u_int64_t)max_lag, data_dir, optimizer.optimizer_,
                    initializer.initializer_, filter.filter_);

  this->embedding_ = std::shared_ptr<Embedding>(empedding_ptr);
}

PyEmbedding::~PyEmbedding() {}

unsigned long long PyEmbedding::lookup(unsigned long long *keys, int len,
                                       Float *data, int n) {
  return (unsigned long long)this->embedding_->lookup(keys, len, data, n);
}
void PyEmbedding::apply_gradients(unsigned long long *keys, int len, Float *gds,
                                  int n, unsigned long long global_step) {
  embedding_->apply_gradients(keys, len, gds, n, global_step);
}
void PyEmbedding::dump(std::string path, int expires) {
  this->embedding_->dump(path, expires);
}