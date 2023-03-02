#include "embedding.h"

Embeddings::Embeddings(int ttl, const std::string &data_dir,
                       const std::shared_ptr<Optimizer> &optimizer,
                       const std::shared_ptr<Initializer> &initializer,
                       const std::shared_ptr<CountingBloomFilter> &filter)
    : db_(nullptr),
      ttl_(ttl),
      optimizer_(optimizer),
      initializer_(initializer),
      filter_(filter) {
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

//更新记录
void Embeddings::update(const u_int64_t &key, MetaData *ptr, Float *gds,
                        const u_int64_t &global_step) {
  const int &dim = this->metas_[groupof(key)].dim;
  ptr->update_num++;
  ptr->update_time = get_current_time();
  this->optimizer_->call(ptr->data, gds, dim, global_step);
}

void Embeddings::lookup(u_int64_t *keys, int len, Float *data, int n) {
  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  std::vector<int> exists(len, -1);
  int j = 0;
  for (int i = 0; i < len; i++) {
    //不配置filter的情况下也可以
    if (this->filter_ == nullptr || this->filter_->check(keys[i])) {
      s_keys.emplace_back(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
      exists[i] = j;
      j++;
    }
  }
  size_t offset = 0;
  MetaData *ptr;

  //写入
  std::vector<rocksdb::Slice> s_put_keys;
  std::vector<std::string> put_result;
  rocksdb::ReadOptions get_options;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  int dim;
  for (int i = 0; i < len; i++) {
    dim = this->metas_[groupof(keys[i])].dim;
    // filter检查没有就置为0
    if (exists[i] == -1) {
      memset(&(data[offset]), 0, sizeof(Float) * dim);
      offset += dim;
      continue;
    }
    j = exists[i];
    if (status[j].ok()) {
      ptr = (MetaData *)&(result[j][0]);
      memcpy(&(data[offset]), ptr->data, sizeof(Float) * dim);
    } else {
      //需要初始化
      std::string value(
          sizeof(MetaData) + sizeof(Float) * this->optimizer_->get_space(dim),
          '\0');
      MetaData *ptr = (MetaData *)(&value[0]);
      this->initializer_->call(ptr->data, dim);
      ptr->update_num = 1;
      ptr->key = keys[i];
      ptr->dim = dim;
      ptr->update_time = get_current_time();
      s_put_keys.emplace_back(
          rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
      put_result.emplace_back(value);
      memcpy(&(data[offset]), ptr->data, sizeof(Float) * dim);
    }
    offset += dim;
  }

  assert(offset == n);

  //写到rocksdb里面去
  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  rocksdb::WriteBatch batch;
  for (size_t i = 0; i < s_put_keys.size(); i++) {
    batch.Put(s_put_keys[i], put_result[i]);
  }
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
    //能够push进去的,必须要filter check ok
    if (this->filter_ == nullptr || this->filter_->check(keys[i])) {
      s_keys.emplace_back(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
      t_gds.push_back(&gds[offset]);
    }
    this->filter_->add(keys[i]);
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
    this->update(meta->key, meta, t_gds[i], global_step);
    batch.Put(s_keys[i], result[i]);
  }
  this->db_->Write(put_options, &batch);
}

//保存
void Embeddings::dump(const std::string &path, int expires) {
  const rocksdb::Snapshot *sp = this->db_->GetSnapshot();
  auto oldest_timestamp = get_current_time() - 86400 * expires;
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

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    ptr = (MetaData *)it->value().data();
    if (ptr->update_time < oldest_timestamp) {
      continue;
    }
    group_counts[groupof(ptr->key)]++;
    writer.write((char *)&ptr->key, sizeof(u_int64_t));
    writer.write((char *)ptr->data, sizeof(Float) * ptr->dim);
  }
  assert(it->status().ok());
  delete it;
  this->db_->ReleaseSnapshot(sp);
  //更新counts
  writer.seekp(sizeof(int) * max_group, std::ios::beg);
  writer.write((char *)&group_counts, sizeof(size_t) * max_group);
  writer.close();
}