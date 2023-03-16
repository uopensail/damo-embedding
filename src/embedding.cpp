#include "embedding.h"

Embeddings::Embeddings(int ttl, int min_count, const std::string &data_dir,
                       const std::shared_ptr<Optimizer> &optimizer,
                       const std::shared_ptr<Initializer> &initializer)
    : db_(nullptr),
      ttl_(ttl),
      min_count_(min_count),
      optimizer_(optimizer),
      initializer_(initializer) {
  rocksdb::Options options;
  options.create_if_missing = true;
  rocksdb::Status status =
      rocksdb::DBWithTTL::Open(options, data_dir, &this->db_, this->ttl_);
  if (!status.ok()) {
    std::cerr << "open leveldb error: " << status.ToString() << std::endl;
    exit(-1);
  }
  assert(this->db_ != nullptr);
  std::cout << "open leveldb: " << data_dir << " successfully!" << std::endl;
}

Embeddings::~Embeddings() { delete this->db_; }

void Embeddings::add_group(int group, int dim) {
  assert(group >= 0 && group < max_group);
  this->metas_[group].dim = dim;
  this->metas_[group].group = group;
}

void Embeddings::update(const u_int64_t &key, MetaData *ptr, Float *gds,
                        const u_int64_t &global_step) {
  const int &dim = this->metas_[groupof(key)].dim;
  ptr->update_num++;
  ptr->update_time = get_current_time();
  this->optimizer_->call(ptr->data, gds, dim, global_step);
}

void Embeddings::create(const u_int64_t &key, MetaData *ptr) {
  const int &dim = this->metas_[groupof(key)].dim;
  this->initializer_->call(ptr->data, dim);
  ptr->update_num = 1;
  ptr->key = key;
  ptr->dim = dim;
  ptr->update_time = get_current_time();
}

void Embeddings::lookup(u_int64_t *keys, int len, Float *data, int n) {
  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  for (int i = 0; i < len; i++) {
    s_keys.emplace_back(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
  }
  rocksdb::ReadOptions get_options;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  size_t offset = 0;
  MetaData *ptr;

  //写入
  rocksdb::WriteBatch batch;
  int dim;
  for (int i = 0; i < len; i++) {
    dim = this->metas_[groupof(keys[i])].dim;
    if (status[i].ok()) {
      ptr = (MetaData *)&(result[i][0]);
      if (ptr->update_num >= this->min_count_) {
        memcpy(&(data[offset]), ptr->data, sizeof(Float) * dim);
      } else {
        memset(&(data[offset]), 0, sizeof(Float) * dim);
      }
    } else {
      //需要初始化
      std::string value(
          sizeof(MetaData) + sizeof(Float) * this->optimizer_->get_space(dim),
          '\0');
      MetaData *ptr = (MetaData *)(&value[0]);
      this->create(keys[i], ptr);
      memset(&(data[offset]), 0, sizeof(Float) * dim);
      batch.Put(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)), value);
    }
    offset += dim;
  }

  assert(offset == n);

  //写到rocksdb里面去
  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  this->db_->Write(put_options, &batch);
  return;
}

void Embeddings::apply_gradients(u_int64_t *keys, int len, Float *gds, int n,
                                 const u_int64_t &global_step) {
  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  std::vector<Float *> t_gds;
  size_t offset = 0;
  for (int i = 0; i < len; i++) {
    s_keys.emplace_back(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
    t_gds.push_back(&gds[offset]);
    offset += this->metas_[groupof(keys[i])].dim;
  }

  //先从rocksdb中进行查找
  rocksdb::ReadOptions get_options;
  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  MetaData *meta = nullptr;
  rocksdb::WriteBatch batch;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  for (size_t i = 0; i < status.size(); i++) {
    if (!status[i].ok()) {
      continue;
    }
    meta = (MetaData *)&(result[i][0]);
    if (meta->update_num < this->min_count_) {
      meta->update_num++;
      meta->update_time = get_current_time();
    } else {
      this->update(meta->key, meta, t_gds[i], global_step);
    }
    batch.Put(s_keys[i], result[i]);
  }
  this->db_->Write(put_options, &batch);
}

//保存
void Embeddings::dump(const std::string &path, int expires) {
  auto oldest_timestamp = get_current_time() - 86400 * expires;
  const rocksdb::Snapshot *sp = this->db_->GetSnapshot();
  rocksdb::ReadOptions read_option;
  read_option.snapshot = sp;
  rocksdb::Iterator *it = this->db_->NewIterator(rocksdb::ReadOptions());
  MetaData *ptr;
  //初始化一些值
  size_t group_counts[max_group];
  int group_dims[max_group];
  for (int i = 0; i < max_group; i++) {
    group_counts[i] = 0;
    group_dims[i] = this->metas_[i].dim;
  }

  std::ofstream writer(path, std::ios::out | std::ios::binary);
  writer.write((char *)&group_dims, sizeof(int) * max_group);
  writer.write((char *)&group_counts, sizeof(size_t) * max_group);
  rocksdb::WriteOptions del_options;
  del_options.sync = false;

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    ptr = (MetaData *)it->value().data();
    if (ptr->update_time < oldest_timestamp ||
        ptr->update_num < this->min_count_) {
      //删除太老或更新次数太少的记录
      this->db_->Delete(del_options, {(char *)&ptr->key, sizeof(u_int64_t)});
      continue;
    }

    group_counts[groupof(ptr->key)]++;
    writer.write((char *)&ptr->key, sizeof(u_int64_t));
    writer.write((char *)ptr->data, sizeof(Float) * ptr->dim);
  }
  assert(it->status().ok());
  delete it;
  this->db_->ReleaseSnapshot(sp);
  // update group key counts
  writer.seekp(sizeof(int) * max_group, std::ios::beg);
  writer.write((char *)&group_counts, sizeof(size_t) * max_group);
  writer.close();

  // do compact
  rocksdb::CompactRangeOptions options;
  this->db_->CompactRange(options, nullptr, nullptr);
}