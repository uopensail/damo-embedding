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

std::string Parameters::to_json() {
  std::string result = "{ ";

  int index = 0;
  for (const auto &kv : *this->params_) {
    if (index != 0) {
      result += ", ";
    }
    result += "\"" + kv.first + "\" : ";
    if (auto v = std::dynamic_pointer_cast<cpptoml::value<double>>(kv.second)) {
      result += std::to_string(static_cast<double>(v->get()));
    } else if (auto v = std::dynamic_pointer_cast<cpptoml::value<std::string>>(
                   kv.second)) {
      result += "\"" + static_cast<std::string>(v->get()) + "\"";
    } else if (auto v = std::dynamic_pointer_cast<cpptoml::value<int64_t>>(
                   kv.second)) {
      result += std::to_string(static_cast<int64_t>(v->get()));
    } else if (auto v =
                   std::dynamic_pointer_cast<cpptoml::value<bool>>(kv.second)) {
      if (static_cast<bool>(v->get())) {
        result += "true";
      } else {
        result += "false";
      }
    }
    index++;
  }
  result += " }";
  return result;
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

void PyInitializer::call(py::array_t<float> w) {
  py::buffer_info info = w.request();
  assert(info.ndim == 1);
  size_t size = info.size;
  float *ptr = static_cast<float *>(info.ptr);
  initializer_->call(ptr, size);
}

PyInitializer::~PyInitializer() {}

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

void PyOptimizer::call(py::array_t<float> w, py::array_t<float> gd,
                       int64_t global_step) {
  py::buffer_info winfo = w.request();
  assert(winfo.ndim == 1);
  size_t wsize = winfo.size;
  float *wptr = static_cast<float *>(winfo.ptr);

  py::buffer_info gdinfo = gd.request();
  assert(gdinfo.ndim == 1);
  size_t gdsize = gdinfo.size;
  float *gdptr = static_cast<float *>(gdinfo.ptr);
  assert(optimizer_->get_space(gdsize) == wsize);
  optimizer_->call(wptr, gdptr, wsize, global_step);
}

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

bool PyFilter::check(int group, int64_t key) {
  if (this->filter_ == nullptr) {
    return true;
  }
  Key x{group, key};
  return filter_->check(x);
}

void PyFilter::add(int group, int64_t key, int64_t num) {
  if (this->filter_ != nullptr) {
    Key x{group, key};
    this->filter_->add(x, num);
  }
}

PyFilter::~PyFilter() {}

PyStorage::PyStorage(const std::string &data_dir, int ttl) {
  this->storage_ = std::make_shared<Storage>(ttl, data_dir);
}

PyStorage::~PyStorage() {}

void PyStorage::checkpoint(const std::string &path) {
  this->storage_->checkpoint(path);
}

void PyStorage::load_from_checkpoint(const std::string &path) {
  this->storage_->load_from_checkpoint(path);
}

void PyStorage::dump(const std::string &path, Parameters condition) {
  auto func = [&condition](MetaData *ptr) -> bool {
    if (ptr == nullptr) {
      return false;
    }
    if (condition.params_ == nullptr) {
      return true;
    }

    Params p(condition.params_);
    if (p.contains("expire_days")) {
      int64_t expire_days = p.get<int64_t>("expire_days");
      auto oldest_ts = get_current_time() - expire_days * 86400000;
      if (ptr->update_time < oldest_ts) {
        return false;
      }
    }

    if (p.contains("min_count")) {
      int64_t min_count = p.get<int64_t>("min_count");
      if (ptr->update_num < min_count) {
        return false;
      }
    }

    if (p.contains("group")) {
      int group = p.get<int>("group");
      if (ptr->group != group) {
        return false;
      }
    }

    return true;
  };
  this->storage_->dump(path, func);
}

void PyStorage::dump(const std::string &path) {
  this->storage_->dump(path, nullptr);
}

PyEmbedding::PyEmbedding(PyStorage storage, PyOptimizer optimizer,
                         PyInitializer initializer, int dim, int group) {
  this->embedding_ =
      std::make_shared<Embedding>(*storage.storage_, optimizer.optimizer_,
                                  initializer.initializer_, dim, group);
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

void PyEmbedding::lookup(py::array_t<int64_t> keys, py::array_t<float> w) {
  py::buffer_info keys_info = keys.request();
  assert(keys_info.ndim == 1);
  int64_t *keys_ptr = static_cast<int64_t *>(keys_info.ptr);

  py::buffer_info w_info = w.request();
  assert(w_info.ndim == 1);
  float *w_ptr = static_cast<float *>(w_info.ptr);
  this->embedding_->lookup(keys_ptr, keys_info.size, w_ptr, w_info.size);
}

void PyEmbedding::apply_gradients(py::array_t<int64_t> keys,
                                  py::array_t<float> gds) {
  py::buffer_info keys_info = keys.request();
  assert(keys_info.ndim == 1);
  int64_t *keys_ptr = static_cast<int64_t *>(keys_info.ptr);

  py::buffer_info gds_info = gds.request();
  assert(gds_info.ndim == 1);
  float *gds_ptr = static_cast<float *>(gds_info.ptr);
  this->embedding_->apply_gradients(keys_ptr, keys_info.size, gds_ptr,
                                    gds_info.size);
}