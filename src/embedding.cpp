#include "embedding.h"

Embedding::Embedding(Storage &storage,
                     const std::shared_ptr<Optimizer> &optimizer,
                     const std::shared_ptr<Initializer> &initializer, int dim,
                     int count)
    : dim_(dim),
      count_(count),
      db_(storage.db_),
      optimizer_(optimizer),
      initializer_(initializer) {}

const int Embedding::get_dim() const { return dim_; }

const u_int64_t Embedding::get_count() const { return this->count_; };

Embedding::~Embedding() {}

std::shared_ptr<std::string> Embedding::create(const u_int64_t &key) {
  auto value = std::make_shared<std::string>(
      sizeof(MetaData) +
          sizeof(Float) * this->optimizer_->get_space(this->dim_),
      0);
  MetaData *ptr = (MetaData *)(value->data());
  this->initializer_->call(ptr->data, this->dim_);
  ptr->update_num = 1;
  ptr->key = key;
  ptr->dim = this->dim_;
  ptr->update_time = get_current_time();
  return value;
}

void Embedding::update(const u_int64_t &key, MetaData *ptr, Float *gds,
                       const u_int64_t &global_step) {
  ptr->update_num++;
  ptr->update_time = get_current_time();
  this->optimizer_->call(ptr->data, gds, this->dim_, global_step);
}

void Embedding::update(const u_int64_t &key, MetaData *ptr) {
  ptr->update_num++;
  ptr->update_time = get_current_time();
}

void Embedding::lookup(u_int64_t *keys, int len, Float *data, int n) {
  assert(len * this->dim_ == n);
  memset(data, 0, n * sizeof(Float));

  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  for (int i = 0; i < len; i++) {
    s_keys.emplace_back(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
  }
  rocksdb::ReadOptions get_options;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  MetaData *ptr;

  rocksdb::WriteBatch batch;
  for (int i = 0; i < len; i++) {
    if (status[i].ok()) {
      ptr = (MetaData *)(result[i].data());
      if (ptr->update_num >= this->count_) {
        memcpy(&(data[i * this->dim_]), ptr->data, sizeof(Float) * this->dim_);
      }
    } else {
      auto value = this->create(keys[i]);
      batch.Put(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)), *value);
    }
  }

  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  this->db_->Write(put_options, &batch);
  return;
}

void Embedding::apply_gradients(u_int64_t *keys, int len, Float *gds, int n,
                                const u_int64_t &global_step) {
  assert(len * this->dim_ == n);
  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  for (int i = 0; i < len; i++) {
    s_keys.emplace_back(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
  }

  // get data from recksdb
  rocksdb::ReadOptions get_options;
  MetaData *ptr = nullptr;
  rocksdb::WriteBatch batch;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  for (int i = 0; i < len; i++) {
    if (!status[i].ok()) {
      continue;
    }
    ptr = (MetaData *)(result[i].data());
    if (ptr->update_num < this->count_) {
      this->update(keys[i], ptr);
    } else {
      this->update(keys[i], ptr, &(gds[i * this->dim_]), global_step);
    }
    batch.Put(s_keys[i], result[i]);
  }
  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  this->db_->Write(put_options, &batch);
}

Storage::Storage(int ttl, const std::string &data_dir) : ttl_(ttl) {
  rocksdb::Options options;
  options.create_if_missing = true;
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
      delete (rocksdb::DBWithTTL *)ptr;
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
  rocksdb::Iterator *it = this->db_->NewIterator(rocksdb::ReadOptions());
  MetaData *ptr;
  size_t group_counts[max_group];
  int group_dims[max_group];
  for (int i = 0; i < max_group; i++) {
    group_counts[i] = 0;
    group_dims[i] = 0;
  }

  std::ofstream writer(path, std::ios::out | std::ios::binary);
  writer.write((char *)&group_dims, sizeof(int) * max_group);
  writer.write((char *)&group_counts, sizeof(size_t) * max_group);
  rocksdb::WriteOptions del_options;
  del_options.sync = false;
  int group;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    ptr = (MetaData *)it->value().data();
    if (filter(ptr)) {
      group = groupof(ptr->key);
      group_counts[group]++;
      group_dims[group] = ptr->dim;
      writer.write((char *)&ptr->key, sizeof(u_int64_t));
      writer.write((char *)ptr->data, sizeof(Float) * ptr->dim);
    }
  }
  assert(it->status().ok());
  delete it;
  this->db_->ReleaseSnapshot(sp);
  // update group key dim and counts
  writer.seekp(0, std::ios::beg);
  writer.write((char *)&group_dims, sizeof(int) * max_group);
  writer.write((char *)&group_counts, sizeof(size_t) * max_group);
  writer.close();

  // do compact
  rocksdb::CompactRangeOptions options;
  this->db_->CompactRange(options, nullptr, nullptr);
}