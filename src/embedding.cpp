#include "embedding.h"

Configure::Configure() {
  dim = 0;
  group = -1;
  optimizer = nullptr;
  initializer = nullptr;
}

GlobalGroupConfigure::GlobalGroupConfigure()
    : configures_(std::make_shared<std::unordered_map<int, Configure>>()) {}

const Configure *GlobalGroupConfigure::operator[](int group) const {
  auto iter = this->configures_->find(group);
  if (iter != this->configures_->end()) {
    return &iter->second;
  }
  return nullptr;
}

void GlobalGroupConfigure::add(int group, const Configure &configure) {
  std::lock_guard<std::mutex> guard(this->group_lock_);
  auto iter = this->configures_->find(group);
  if (iter == this->configures_->end()) {
    auto configures = std::make_shared<std::unordered_map<int, Configure>>();
    for (auto &config : *this->configures_) {
      configures->insert(std::make_pair(config.first, config.second));
    }
    configures->insert(std::make_pair(group, configure));
    this->configures_.swap(configures);
  } else {
    std::cerr << "group: " << group << " exists" << std::endl;
    exit(-1);
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
  auto cfg = global_groiup_configure[ptr->group];
  if (ptr->group < 0 || cfg == nullptr) {
    return false;
  }
  assert(new_value != nullptr);
  new_value->resize(existing_value->size());
  MetaData *new_ptr = (MetaData *)(new_value->data());
  memcpy(new_ptr, ptr, existing_value->size());
  for (const auto &value : operand_list) {
    new_ptr->update_num++;
    float *gds = (float *)(const_cast<char *>(value.data()));
    cfg->optimizer->call(new_ptr->data, gds, new_ptr->dim, new_ptr->update_num);
  }
  new_ptr->update_time = get_current_time();
  return true;
}

Embedding::Embedding(Storage &storage,
                     const std::shared_ptr<Optimizer> &optimizer,
                     const std::shared_ptr<Initializer> &initializer, int dim,
                     int group)
    : dim_(dim), group_(group), db_(storage.db_), optimizer_(optimizer),

      initializer_(initializer) {
  if (group < 0) {
    std::cerr << "group: " << group << " out of range" << std::endl;
    exit(-1);
  }

  Configure cfg;
  cfg.dim = dim;
  cfg.group = group;
  cfg.optimizer = optimizer;
  cfg.initializer = initializer;
  global_groiup_configure.add(group, cfg);
}

Embedding::~Embedding() {}

std::shared_ptr<std::string> Embedding::create(const int64_t &key) {
  auto value = std::make_shared<std::string>(
      sizeof(MetaData) +
          sizeof(Float) * this->optimizer_->get_space(this->dim_),
      0);
  MetaData *ptr = (MetaData *)(value->data());
  this->initializer_->call(ptr->data, this->dim_);
  ptr->update_num = 0;
  ptr->key = key;
  ptr->group = this->group_;
  ptr->dim = this->dim_;
  ptr->update_time = get_current_time();
  return value;
}

void Embedding::lookup(int64_t *keys, int len, Float *data, int n) {
  assert(len * this->dim_ == n);
  memset(data, 0, n * sizeof(Float));

  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  Key *group_keys = (Key *)malloc(len * sizeof(Key));
  for (int i = 0; i < len; i++) {
    group_keys[i].group = this->group_;
    group_keys[i].key = keys[i];
    s_keys.emplace_back(rocksdb::Slice((char *)&group_keys[i], sizeof(Key)));
  }
  rocksdb::ReadOptions get_options;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  MetaData *ptr;

  rocksdb::WriteBatch batch;
  for (int i = 0; i < len; i++) {
    if (status[i].ok()) {
      ptr = (MetaData *)(result[i].data());
      memcpy(&(data[i * this->dim_]), ptr->data, sizeof(Float) * this->dim_);
    } else {
      auto value = this->create(keys[i]);
      ptr = (MetaData *)(value->data());
      memcpy(&(data[i * this->dim_]), ptr->data, sizeof(Float) * this->dim_);
      batch.Put(rocksdb::Slice((char *)&group_keys[i], sizeof(Key)), *value);
    }
  }

  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  this->db_->Write(put_options, &batch);
  free(group_keys);
  return;
}

void Embedding::apply_gradients(int64_t *keys, int len, Float *gds, int n) {
  assert(len * this->dim_ == n);
  Key *group_keys = (Key *)malloc(len * sizeof(Key));

  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  rocksdb::WriteBatch batch;

  for (int i = 0; i < len; i++) {
    group_keys[i].group = this->group_;
    group_keys[i].key = keys[i];
    batch.Merge(rocksdb::Slice((char *)&group_keys[i], sizeof(Key)),
                rocksdb::Slice((char *)&gds[i * this->dim_],
                               sizeof(Float) * this->dim_));
  }
  this->db_->Write(put_options, &batch);
  free(group_keys);
}

Storage::Storage(int ttl, const std::string &data_dir) : ttl_(ttl) {
  rocksdb::Options options;
  options.create_if_missing = true;
  options.merge_operator.reset(new ApplyGredientsOperator());
  rocksdb::DBWithTTL *db;
  rocksdb::Status status =
      rocksdb::DBWithTTL::Open(options, data_dir, &db, this->ttl_);
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
    }
  });

  std::cout << "open leveldb: " << data_dir << " successfully!" << std::endl;
}

Storage::~Storage() {}

void Storage::dump(const std::string &path,
                   const std::function<bool(MetaData *ptr)> &filter) {
  const rocksdb::Snapshot *sp = this->db_->GetSnapshot();
  rocksdb::ReadOptions read_option;
  read_option.snapshot = sp;
  rocksdb::Iterator *it = this->db_->NewIterator(read_option);
  MetaData *ptr;
  int size = global_groiup_configure.configures_->size();
  int *group = (int *)calloc(size, sizeof(int));
  int *group_dims = (int *)calloc(size, sizeof(int));
  std::unordered_map<int, int> group_index;
  int index = 0;
  for (auto &ptr : *global_groiup_configure.configures_) {
    group[index] = ptr.first;
    group_dims[index] = ptr.second.dim;
    group_index[ptr.first] = index;
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
    if (filter == nullptr || filter(ptr)) {
      group_counts[group_index[ptr->group]]++;
      writer.write((char *)&ptr->key, sizeof(int64_t));
      writer.write((char *)&ptr->group, sizeof(int));
      writer.write((char *)ptr->data, sizeof(Float) * ptr->dim);
    }
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
void Storage::checkpoint(const std::string &path) {
  std::string checkpoint_path = path + "-" + std::to_string(get_current_time());
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

void Storage::load_from_checkpoint(const std::string &path) {
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
}