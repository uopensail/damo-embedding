#include "damo.h"

PyDamo::PyDamo(const std::string &config_file) {
  std::ifstream file(config_file);
  if (!file) {
    std::cerr << "Failed to open JSON file." << std::endl;
    exit(-1);
  }
  json configure;
  try {
    file >> configure;
  } catch (const std::exception &e) {
    std::cerr << "JSON parsing error: " << e.what() << std::endl;
    exit(-1);
  }
  file.close();

  this->warehouse_ = std::make_shared<EmbeddingWareHouse>(configure);
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

void PyDamo::push(uint64_t step_control, int group, py::array_t<int64_t> keys,
                  py::array_t<float> gds) {
  py::buffer_info keys_info = keys.request();
  assert(keys_info.ndim == 1);
  int64_t *keys_ptr = static_cast<int64_t *>(keys_info.ptr);

  py::buffer_info gds_info = gds.request();
  assert(gds_info.ndim == 1);
  float *gds_ptr = static_cast<float *>(gds_info.ptr);
  this->warehouse_->apply_gradients(step_control, group, keys_ptr, keys_info.size, gds_ptr,
                                    gds_info.size);
}

std::string PyDamo::to_json() { return this->warehouse_->to_json(); }