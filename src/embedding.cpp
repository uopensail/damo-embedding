#include "embedding.h"

Embedding::Embedding(json &p) {
  assert(p.contains("initializer"));
  assert(p.contains("optimizer"));
  assert(p.contains("dim"));
  assert(p.contains("group"));

  this->dim = p["dim"].get<int>();
  this->group = p["group"].get<int>();
  assert(0 <= this->group && this->group < max_embedding_num);

  this->initializer = get_initializers(Params{p["initializer"]});
  this->optimizer = get_optimizers(Params{p["optimizer"]});
}

Embedding::~Embedding() {}

ApplyGredientsOperator::ApplyGredientsOperator(json &configure) {
  assert(configure.contains("embeddings"));
  assert(configure["embeddings"].is_array());
  for (int i = 0; i < max_embedding_num; i++) {
    this->embeddings_[i] = nullptr;
  }
  for (auto &e : configure["embeddings"]) {
    auto embedding = std::make_shared<Embedding>(e);
    this->embeddings_[embedding->group] = embedding;
  }
}

bool ApplyGredientsOperator::FullMerge(
    const rocksdb::Slice &key, const rocksdb::Slice *existing_value,
    const std::deque<std::string> &operand_list, std::string *new_value,
    rocksdb::Logger *logger) const {
  // key must already exist
  if (existing_value == nullptr) {
    return false;
  }

  MetaData *ptr = (MetaData *)(const_cast<char *>(existing_value->data()));
  auto embedding = this->embeddings_[ptr->group];
  if (ptr->group < 0 || embedding == nullptr) {
    return false;
  }
  assert(new_value != nullptr);
  new_value->resize(existing_value->size());
  MetaData *new_ptr = (MetaData *)(new_value->data());
  memcpy(new_ptr, ptr, existing_value->size());
  for (const auto &value : operand_list) {
    new_ptr->update_num++;
    uint64_t step_control = *((uint64_t*)value.data());
    if (step_control != 0 
        && ((step_control&0xffffffff00000000) <= (new_ptr->step_control&0xffffffff00000000))
        && ((step_control&0x00000000ffffffff) <= (new_ptr->step_control&0x00000000ffffffff))) {
          continue;
    }
    float *gds = (float *)((const_cast<char *>(value.data()))+STEP_CONTROL_BYTESIZE);
    embedding->optimizer->call(new_ptr->data, gds, new_ptr->dim,
                               new_ptr->update_num);
  }
  new_ptr->update_time = get_current_time();
  return true;
}

EmbeddingWareHouse::EmbeddingWareHouse(json &configure)
    : configure_(configure) {
  // create embeddings
  assert(configure_.contains("embeddings"));
  for (int i = 0; i < max_embedding_num; i++) {
    this->embeddings_[i] = nullptr;
  }
  assert(configure_["embeddings"].is_array());
  this->size_ = configure_["embeddings"].size();
  for (auto &e : configure_["embeddings"]) {
    auto embedding = std::make_shared<Embedding>(e);
    this->embeddings_[embedding->group] = embedding;
  }

  // open rocksdb
  assert(configure_.contains("ttl"));
  assert(configure_.contains("dir"));
  int ttl = configure_["ttl"].get<int>();
  std::string dir = configure_["dir"].get<std::string>();

  rocksdb::Options options;
  options.create_if_missing = true;
  options.merge_operator.reset(new ApplyGredientsOperator(configure_));
  rocksdb::DBWithTTL *db;
  rocksdb::Status status = rocksdb::DBWithTTL::Open(options, dir, &db, ttl);
  if (!status.ok()) {
    std::cerr << "open leveldb error: " << status.ToString() << std::endl;
    exit(-1);
  }
  assert(db != nullptr);

  this->db_ = std::shared_ptr<rocksdb::DBWithTTL>(db, [](void *ptr) {
    if (ptr != nullptr) {
      rocksdb::DBWithTTL *db = (rocksdb::DBWithTTL *)ptr;
      db->Flush(rocksdb::FlushOptions());
      // do compact
      db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
      db->Close();
      delete db;
      db = nullptr;
      std::cout << "rocksdb closed!" << std::endl;
    }
  });

  std::cout << "open leveldb: " << dir << " successfully!" << std::endl;

  if (configure_.contains("reload_dir")) {
    std::string reload_dir = configure_["reload_dir"].get<std::string>();
    this->load(reload_dir);
  }
}

int EmbeddingWareHouse::dim(int group) const {
  assert(0 <= group && group < max_embedding_num);
  return embeddings_[group]->dim;
}

std::string EmbeddingWareHouse::to_json() { return this->configure_.dump(); }

void EmbeddingWareHouse::dump(const std::string &path) {
  const rocksdb::Snapshot *sp = this->db_->GetSnapshot();
  rocksdb::ReadOptions read_option;
  read_option.snapshot = sp;
  rocksdb::Iterator *it = this->db_->NewIterator(read_option);
  MetaData *ptr;
  int size = this->size_;
  int *group = (int *)calloc(size, sizeof(int));
  int *group_dims = (int *)calloc(size, sizeof(int));
  std::unordered_map<int, int> group_index;
  int index = 0;
  for (int i = 0; i < max_embedding_num; i++) {
    auto embedding = this->embeddings_[i];
    if (embedding == nullptr) {
      continue;
    }
    group[index] = embedding->group;
    group_dims[index] = embedding->dim;
    group_index[embedding->group] = index;
    index++;
  }
  int64_t *group_counts = (int64_t *)calloc(size, sizeof(int64_t));

  std::ofstream writer(path, std::ios::out | std::ios::binary);
  writer.write((char *)&size, sizeof(int));
  writer.write((char *)group, sizeof(int) * size);
  writer.write((char *)group_dims, sizeof(int) * size);
  writer.write((char *)group_counts, sizeof(int64_t) * size);
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    ptr = (MetaData *)it->value().data();
    if (ptr->key == 0) {
      continue;
    }
    group_counts[group_index[ptr->group]]++;
    writer.write((char *)&ptr->key, sizeof(int64_t));
    writer.write((char *)&ptr->group, sizeof(int));
    writer.write((char *)ptr->data, sizeof(Float) * ptr->dim);
  }
  assert(it->status().ok());
  delete it;
  this->db_->ReleaseSnapshot(sp);
  // update group counts
  writer.seekp(sizeof(int) * (1 + size * 2), std::ios::beg);
  writer.write((char *)group_counts, sizeof(size_t) * size);
  writer.close();
}

// checkpoint file format:
// int64_t: key count
// (size_t: key length, bytes: key data, size_t: value length, bytes: value
// data)+
void EmbeddingWareHouse::checkpoint(const std::string &checkpoint_path) {
  const rocksdb::Snapshot *sp = this->db_->GetSnapshot();
  rocksdb::ReadOptions read_option;
  read_option.snapshot = sp;
  rocksdb::Iterator *it = this->db_->NewIterator(read_option);
  int64_t count = 0;
  size_t key_len = 0, value_len = 0;

  std::ofstream writer(checkpoint_path, std::ios::out | std::ios::binary);
  writer.write((char *)&count, sizeof(int64_t));

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    count++;
    key_len = it->key().size();
    value_len = it->value().size();
    writer.write((char *)&key_len, sizeof(size_t));
    writer.write(it->key().data(), key_len);
    writer.write((char *)&value_len, sizeof(size_t));
    writer.write(it->value().data(), value_len);
  }
  assert(it->status().ok());
  delete it;
  this->db_->ReleaseSnapshot(sp);

  writer.seekp(0, std::ios::beg);
  writer.write((char *)&count, sizeof(int64_t));
  writer.close();
}

void EmbeddingWareHouse::load(const std::string &path) {
  std::cout << "loading checkpoint from:" << path << std::endl;
  // first delete all the old keys
  auto status = this->db_->DeleteRange(rocksdb::WriteOptions(),
                                       this->db_->DefaultColumnFamily(),
                                       rocksdb::Slice(), rocksdb::Slice());

  // add keys read from checkpoint file
  rocksdb::WriteOptions options;
  options.sync = false;
  std::ifstream reader(path, std::ios::in | std::ios::binary);
  int64_t count = 0;
  size_t key_len = 0, value_len = 0;
  reader.read((char *)&count, sizeof(int64_t));
  size_t max_key_length = 1024, max_value_length = 1024;
  char *key = (char *)malloc(max_key_length);
  char *value = (char *)malloc(max_value_length);

  for (int64_t i = 0; i < count; i++) {
    reader.read((char *)&key_len, sizeof(size_t));
    if (key_len > max_key_length) {
      max_key_length = key_len * 2;
      free(key);
      key = (char *)malloc(max_key_length);
    }
    reader.read(key, key_len);
    reader.read((char *)&value_len, sizeof(size_t));
    if (value_len > max_value_length) {
      max_value_length = value_len * 2;
      free(value);
      value = (char *)malloc(max_value_length);
    }
    reader.read(value, value_len);
    this->db_->Put(options, {key, key_len}, {value, value_len});
  }
  free(key);
  free(value);
  reader.close();
  this->db_->Flush(rocksdb::FlushOptions());
  this->db_->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
  std::cout << "finish loading checkpoint!" << std::endl;
}

std::shared_ptr<std::string>
EmbeddingWareHouse::create_record(int group, const int64_t &key) {
  assert(0 <= group && group < max_embedding_num);
  auto embedding = this->embeddings_[group];
  assert(embedding != nullptr);
  int dim = embedding->dim;

  auto value = std::make_shared<std::string>(
      sizeof(MetaData) + sizeof(Float) * embedding->optimizer->get_space(dim),
      0);
  MetaData *ptr = (MetaData *)(value->data());
  embedding->initializer->call(ptr->data, dim);
  ptr->update_num = 0;
  ptr->key = key;
  ptr->group = group;
  ptr->dim = dim;
  ptr->update_time = get_current_time();
  return value;
}

void EmbeddingWareHouse::lookup(int group, int64_t *keys, int len, Float *data,
                                int n) {
  assert(0 <= group && group < max_embedding_num);
  auto embedding = this->embeddings_[group];
  assert(embedding != nullptr);
  int dim = embedding->dim;
  assert(len * dim == n);

  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  Key *group_keys = (Key *)malloc(len * sizeof(Key));
  for (int i = 0; i < len; i++) {
    if (keys[i] == 0) {
      continue;
    }
    group_keys[i].group = group;
    group_keys[i].key = keys[i];
    s_keys.emplace_back(rocksdb::Slice((char *)&group_keys[i], sizeof(Key)));
  }
  rocksdb::ReadOptions get_options;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  MetaData *ptr;

  rocksdb::WriteBatch batch;
  int index = 0;
  for (int i = 0; i < len; i++) {
    // filter 0
    if (keys[i] == 0) {
      continue;
    }
    if (status[index].ok()) {
      ptr = (MetaData *)(result[index].data());
      memcpy(&(data[i * dim]), ptr->data, sizeof(Float) * dim);
    } else {
      auto value = this->create_record(group, keys[i]);
      ptr = (MetaData *)(value->data());
      memcpy(&(data[i * dim]), ptr->data, sizeof(Float) * dim);
      batch.Put(rocksdb::Slice((char *)&group_keys[i], sizeof(Key)), *value);
    }
    index++;
  }

  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  this->db_->Write(put_options, &batch);
  free(group_keys);
  return;
}

void EmbeddingWareHouse::apply_gradients(uint64_t step_control, int group, int64_t *keys, int len,
                                         Float *gds, int n) {
  assert(0 <= group && group < max_embedding_num);
  auto embedding = this->embeddings_[group];
  assert(embedding != nullptr);
  int dim = embedding->dim;
  assert(len * dim == n);
  Key *group_keys = (Key *)malloc(len * sizeof(Key));

  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  rocksdb::WriteBatch batch;
  char* tmp_value = (char*)malloc(STEP_CONTROL_BYTESIZE + sizeof(Float) * dim);
  for (int i = 0; i < len; i++) {
    // filter 0
    if (keys[i] == 0) {
      continue;
    }
    group_keys[i].group = group;
    group_keys[i].key = keys[i];
    memcpy(tmp_value, &step_control, STEP_CONTROL_BYTESIZE);
    memcpy(tmp_value+STEP_CONTROL_BYTESIZE, (char *)&gds[i * dim],sizeof(Float) * dim);
    batch.Merge(rocksdb::Slice((char *)&group_keys[i], sizeof(Key)),
                rocksdb::Slice(tmp_value, STEP_CONTROL_BYTESIZE + sizeof(Float) * dim));
  }
  this->db_->Write(put_options, &batch);
  free(tmp_value);
  free(group_keys);
}