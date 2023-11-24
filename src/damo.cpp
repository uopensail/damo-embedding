#include "damo.h"

void damo_open(int ttl, const std::string &dir) {
  global_embedding_warehouse->opendb(ttl, dir);
}

void damo_new(const std::string &params) {
  json p = json::parse(params);
  global_embedding_warehouse->insert(p);
}

void damo_dump(const std::string &dir) {
  global_embedding_warehouse->dump(dir);
}

void damo_checkpoint(const std::string &dir) {
  global_embedding_warehouse->checkpoint(dir);
}

void damo_load(const std::string &dir) {
  global_embedding_warehouse->load(dir);
}

void damo_pull(int group, py::array_t<int64_t> keys, py::array_t<float> w) {
  py::buffer_info keys_info = keys.request();
  assert(keys_info.ndim == 1);
  int64_t *keys_ptr = static_cast<int64_t *>(keys_info.ptr);

  py::buffer_info w_info = w.request();
  assert(w_info.ndim == 1);
  float *w_ptr = static_cast<float *>(w_info.ptr);
  global_embedding_warehouse->lookup(group, keys_ptr, keys_info.size, w_ptr,
                                     w_info.size);
}

void damo_push(int group, py::array_t<int64_t> keys, py::array_t<float> gds) {
  py::buffer_info keys_info = keys.request();
  assert(keys_info.ndim == 1);
  int64_t *keys_ptr = static_cast<int64_t *>(keys_info.ptr);

  py::buffer_info gds_info = gds.request();
  assert(gds_info.ndim == 1);
  float *gds_ptr = static_cast<float *>(gds_info.ptr);
  global_embedding_warehouse->lookup(group, keys_ptr, keys_info.size, gds_ptr,
                                     gds_info.size);
}
