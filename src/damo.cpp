#include "damo.h"

PyDamo::PyDamo(const std::string &config_file) {
  this->warehouse_ = std::make_shared<EmbeddingWareHouse>(config_file);
}

void PyDamo::dump(const std::string &dir) { this->warehouse_->dump(dir); }

void PyDamo::checkpoint(const std::string &dir) {
  this->warehouse_->checkpoint(dir);
}

// void PyDamo::load(const std::string &dir) { this->warehouse_->load(dir); }

void PyDamo::pull(int group, py::array_t<int64_t> keys, py::array_t<float> w) {
  py::buffer_info keys_info = keys.request();
  assert(keys_info.ndim == 1);
  int64_t *keys_ptr = static_cast<int64_t *>(keys_info.ptr);

  py::buffer_info w_info = w.request();
  assert(w_info.ndim == 1);
  float *w_ptr = static_cast<float *>(w_info.ptr);
  this->warehouse_->lookup(group, keys_ptr, keys_info.size, w_ptr, w_info.size);
}

void PyDamo::push(int group, py::array_t<int64_t> keys,
                  py::array_t<float> gds) {
  py::buffer_info keys_info = keys.request();
  assert(keys_info.ndim == 1);
  int64_t *keys_ptr = static_cast<int64_t *>(keys_info.ptr);

  py::buffer_info gds_info = gds.request();
  assert(gds_info.ndim == 1);
  float *gds_ptr = static_cast<float *>(gds_info.ptr);
  this->warehouse_->apply_gradients(group, keys_ptr, keys_info.size, gds_ptr,
                                    gds_info.size);
}

std::string PyDamo::to_json() { return this->warehouse_->to_json(); }