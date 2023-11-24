#include "damo-embedding.h"
#include "embedding.h"

void damo_embedding_open(int ttl, char *path, int len) {
  std::string dir(path, len);
  global_embedding_warehouse->opendb(ttl, dir);
}

void damo_embedding_close() { global_embedding_warehouse->closedb(); }

void damo_embedding_new(char *data, int len) {
  std::string param_str(data, len);
  json p = json::parse(param_str);
  global_embedding_warehouse->insert(p);
}

void damo_embedding_dump(char *path, int len) {
  std::string dir(path, len);
  global_embedding_warehouse->dump(dir);
}
void damo_embedding_checkpoint(char *path, int len) {
  std::string dir(path, len);
  global_embedding_warehouse->checkpoint(dir);
}
void damo_embedding_load(char *path, int len) {
  std::string dir(path, len);
  global_embedding_warehouse->load(dir);
}

void damo_embedding_pull(int group, void *keys, int klen, void *w, int wlen) {
  global_embedding_warehouse->lookup(group, (int64_t *)keys, klen, (Float *)w,
                                     wlen);
}

void damo_embedding_push(int group, void *keys, int klen, void *g, int glen) {
  global_embedding_warehouse->apply_gradients(group, (int64_t *)keys, klen,
                                              (Float *)g, glen);
}