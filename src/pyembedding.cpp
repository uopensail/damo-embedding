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

PyEmbeddingFactory::PyEmbeddingFactory() {
  PyOptimizer default_optimizer;
  PyInitializer default_initializer;
  this->embeddings_ = std::make_shared<Embeddings>(
      864000, 15, "/tmp/pyembedding", default_optimizer.optimizer_,
      default_initializer.initializer_);
}

PyEmbeddingFactory::PyEmbeddingFactory(const std::string &config_file) {
  std::shared_ptr<cpptoml::table> g = cpptoml::parse_file(config_file);
  Params p_optimizer(g->get_table("optimizer"));
  Params p_initializer(g->get_table("initializer"));
  Params p_storage(g->get_table("storage"));

  std::shared_ptr<Optimizer> optimizer = nullptr;
  Params p_scheduler(g->contains("scheduler") ? g->get_table("scheduler")
                                              : nullptr);

  this->embeddings_ = std::make_shared<Embeddings>(
      p_storage.get<int>("ttl", 864000), p_storage.get<int>("min_count", 15),
      p_storage.get<std::string>("path", "/tmp/pyembedding"),
      get_optimizers(p_optimizer, p_scheduler),
      get_initializers(p_initializer));
}

PyEmbeddingFactory::PyEmbeddingFactory(int ttl, int min_count,
                                       const std::string &data_dir,
                                       PyOptimizer optimizer,
                                       PyInitializer initializer) {
  this->embeddings_ = std::make_shared<Embeddings>(
      ttl, min_count, data_dir, optimizer.optimizer_, initializer.initializer_);
}

PyEmbeddingFactory::~PyEmbeddingFactory() {}

void PyEmbeddingFactory::dump(const std::string &path, int expires) {
  this->embeddings_->dump(path, expires);
}

PyEmbedding::PyEmbedding(PyEmbeddingFactory factory, int group, int dim)
    : embeddings_(factory.embeddings_), group_(group), dim_(dim) {
  this->embeddings_->add_group(group, dim);
}
PyEmbedding::PyEmbedding(const PyEmbedding &p) {
  if (&p != this) {
    this->embeddings_ = p.embeddings_;
    this->group_ = p.group_;
    this->dim_ = p.dim_;
  }
}

PyEmbedding &PyEmbedding::operator=(const PyEmbedding &p) {
  if (&p != this) {
    this->embeddings_ = p.embeddings_;
    this->group_ = p.group_;
    this->dim_ = p.dim_;
  }
  return *this;
}

PyEmbedding::~PyEmbedding() {}

void PyEmbedding::lookup(unsigned long long *keys, int len, float *data,
                         int n) {
  this->embeddings_->lookup((u_int64_t *)keys, len, data, n);
}

void PyEmbedding::apply_gradients(unsigned long long *keys, int len, float *gds,
                                  int n, unsigned long long global_step) {
  this->embeddings_->apply_gradients((u_int64_t *)keys, len, gds, n,
                                     global_step);
}
