#include "embedding.h"

Embeddings::Embeddings(u_int64_t step_lag, int ttl, std::string data_dir,
                       const std::shared_ptr<Optimizer> &optimizer,
                       const std::shared_ptr<Initializer> &initializer,
                       const std::shared_ptr<CountBloomFilter> &filter)
    : db_(nullptr),
      lag_(step_lag),
      ttl_(ttl),
      optimizer_(optimizer),
      initializer_(initializer),
      filter_(filter) {
  rocksdb::Options options;
  options.create_if_missing = true;
  rocksdb::Status status =
      rocksdb::DBWithTTL::Open(options, data_dir, &this->db_, this->ttl_);

  // rocksdb::Status status = rocksdb::DB::Open(options, data_dir, &this->db_);
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

//创建一条记录
std::string *Embeddings::create(u_int64_t &key) {
  const int &dim = this->metas_[groupof(key)].dim;
  int size =
      sizeof(MetaData) + sizeof(Float) * this->optimizer_->get_space(dim);
  std::string *value = new std::string(size, '\0');
  MetaData *ptr = (MetaData *)&((*value)[0]);
  this->initializer_->call(ptr->data, dim);
  ptr->update_num = 1;
  ptr->update_logic_time = 1;
  ptr->key = key;
  ptr->dim = dim;
  ptr->update_real_time = get_current_time();
  this->optimizer_->init_helper(ptr->data, dim);
  return value;
}

//更新记录
void Embeddings::update(u_int64_t &key, MetaData *ptr, Float *gds,
                        u_int64_t global_step) {
  const int &dim = this->metas_[groupof(key)].dim;
  //太滞后的数据就抛掉
  if (ptr->update_logic_time > this->lag_ + global_step) {
    return;
  }
  ptr->update_logic_time = std::max(ptr->update_logic_time, global_step);
  ptr->update_num++;
  ptr->update_real_time = get_current_time();
  this->optimizer_->call(ptr->data, gds, dim, ptr->update_logic_time);
}

u_int64_t Embeddings::lookup(u_int64_t *keys, int len, Float *data, int n) {
  //在lookup的时候不加锁
  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  std::vector<int> exists(len);
  u_int64_t global_step = 1;
  int j = 0;
  for (int i = 0; i < len; i++) {
    //不配置filter的情况下也可以
    if (this->filter_ == nullptr || this->filter_->check(keys[i])) {
      s_keys.emplace_back(rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
      exists[i] = j;
      j++;
    } else {
      exists[i] = -1;
    }
  }
  size_t offset = 0;
  MetaData *ptr;

  //写入
  std::vector<rocksdb::Slice> s_put_keys;
  std::vector<std::string> put_result;
  rocksdb::ReadOptions get_options;
  auto status = this->db_->MultiGet(get_options, s_keys, &result);
  for (int i = 0; i < len; i++) {
    // filter检查没有就置为0
    if (exists[i] == -1) {
      memset(&(data[offset]), 0,
             sizeof(Float) * this->metas_[groupof(keys[i])].dim);
      offset += this->metas_[groupof(keys[i])].dim;
      continue;
    }
    j = exists[i];
    if (status[j].ok()) {
      ptr = (MetaData *)&(result[j][0]);
      memcpy(&(data[offset]), ptr->data,
             sizeof(Float) * this->metas_[groupof(keys[i])].dim);
      global_step = std::max(global_step, ptr->update_logic_time);
    } else {
      //需要初始化
      std::string *tmpValue = this->create(keys[i]);
      s_put_keys.emplace_back(
          rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
      put_result.emplace_back(*tmpValue);
      ptr = (MetaData *)&((*tmpValue)[0]);
      memcpy(&(data[offset]), ptr->data,
             sizeof(Float) * this->metas_[groupof(keys[i])].dim);
      delete tmpValue;
    }
    offset += this->metas_[groupof(keys[i])].dim;
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
  return global_step;
}

void Embeddings::apply_gradients(u_int64_t *keys, int len, Float *gds, int n,
                                 u_int64_t global_step) {
  //写入的时候需要加锁,将key按照锁进行分组
  std::vector<rocksdb::Slice> s_keys_per_lock[max_lock_num];  //分组的key
  std::vector<Float *> gds_per_lock[max_lock_num];            //分组的gd
  int locker_id;
  for (int i = 0; i < len; i++) {
    //能够push进去的,必须要filter check ok
    if (this->filter_ == nullptr || this->filter_->check(keys[i])) {
      locker_id = keys[i] & (max_lock_num - 1);
      gds_per_lock[locker_id].emplace_back(
          &(gds[i * this->metas_[groupof(keys[i])].dim]));
      s_keys_per_lock[locker_id].emplace_back(
          rocksdb::Slice((char *)&keys[i], sizeof(u_int64_t)));
    } else {
      this->filter_->add(keys[i]);
    }
  }

  //先从rocksdb中进行查找
  rocksdb::ReadOptions get_options;
  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  std::vector<std::string> result;
  rocksdb::WriteBatch batch;
  for (int i = 0; i < max_lock_num; i++) {
    if (s_keys_per_lock[i].size() == 0) {
      continue;
    }

    //加锁
    result.clear();
    batch.Clear();
    lockers_[i].lock();
    auto status = this->db_->MultiGet(get_options, s_keys_per_lock[i], &result);
    //遍历所有,更新值
    for (size_t j = 0; j < s_keys_per_lock[i].size(); j++) {
      if (status[j].ok()) {
        //更新数据
        this->update(*(u_int64_t *)(s_keys_per_lock[i][j].data()),
                     (MetaData *)&(result[j][0]), gds_per_lock[i][j],
                     global_step);
        batch.Put(s_keys_per_lock[i][j], result[i]);
      }
    }
    //写回到rocksdb
    this->db_->Write(put_options, &batch);
    lockers_[i].unlock();
    s_keys_per_lock[i].clear();
    gds_per_lock[i].clear();
  }
}

//保存
void Embeddings::dump(std::string path, int expires) {
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
    if (ptr->update_real_time < oldest_timestamp) {
      continue;
    }
    group_counts[groupof(ptr->key)]++;
    writer.write((char *)&ptr->key, sizeof(u_int64_t));
    writer.write((char *)ptr->data, sizeof(Float) * ptr->dim);
  }

  this->db_->ReleaseSnapshot(sp);
  //更新counts
  writer.seekp(sizeof(int) * max_group, std::ios::beg);
  writer.write((char *)&group_counts, sizeof(size_t) * max_group);
  writer.close();
}