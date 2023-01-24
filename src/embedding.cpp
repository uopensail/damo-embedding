#include "embedding.h"

Embedding::Embedding(u_int64_t group, int dim, u_int64_t step_lag,
                     std::string data_dir,
                     const std::shared_ptr<Optimizer> &optimizer,
                     const std::shared_ptr<Initializer> &initializer,
                     const std::shared_ptr<CountBloomFilter> &filter)
    : db_(nullptr), group_(group), dim_(dim), lag_(step_lag),
      optimizer_(optimizer), initializer_(initializer), filter_(filter) {
  rocksdb::Options options;
  options.create_if_missing = true;
  rocksdb::Status status = rocksdb::DB::Open(options, data_dir, &this->db_);
  if (!status.ok()) {
    std::cerr << "open leveldb error: " << status.ToString() << std::endl;
    exit(-1);
  }
  assert(this->db_ != nullptr);
  std::cout << "open leveldb: " << data_dir << " successfully!" << std::endl;
}

Embedding::~Embedding() { delete this->db_; }

//创建一条记录
std::string *Embedding::create(u_int64_t &key) {
  int size = sizeof(MetaData) +
             sizeof(Float) * this->optimizer_->get_space(this->dim_);
  std::string *value = new std::string(size, '\0');
  MetaData *ptr = (MetaData *)&((*value)[0]);
  this->initializer_->call(ptr->data, this->dim_);
  ptr->update_num = 1;
  ptr->update_logic_time = 1;
  ptr->key = mask_group(this->group_, key);
  ptr->dim = dim_;
  ptr->update_real_time = get_current_time();
  this->optimizer_->init_helper(ptr->data, this->dim_);
  return value;
}

//更新记录
void Embedding::update(MetaData *ptr, Float *gds, u_int64_t global_step) {
  //太滞后的数据就抛掉
  if (ptr->update_logic_time > this->lag_ + global_step) {
    return;
  }
  ptr->update_logic_time = std::max(ptr->update_logic_time, global_step);
  ptr->update_num++;
  ptr->update_real_time = get_current_time();
  this->optimizer_->call(ptr->data, gds, this->dim_, ptr->update_logic_time);
}

u_int64_t Embedding::lookup(u_int64_t *keys, int len, Float *data, int n) {
  //在lookup的时候不加锁
  std::vector<rocksdb::Slice> s_keys;
  std::vector<std::string> result;
  std::vector<int> exists(len);
  u_int64_t global_step = 1;
  int j = 0;
  u_int64_t key;
  for (int i = 0; i < len; i++) {
    key = mask_group(this->group_, keys[i]);
    //不配置filter的情况下也可以
    if (filter_ == nullptr || filter_->check(key)) {
      s_keys.emplace_back(rocksdb::Slice((char *)&key, sizeof(u_int64_t)));
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
  auto status = db_->MultiGet(get_options, s_keys, &result);
  for (int i = 0; i < len; i++) {
    // filter检查没有就置为0
    if (exists[i] == -1) {
      memset(&(data[offset]), 0, sizeof(Float) * this->dim_);
      offset += this->dim_;
      continue;
    }
    j = exists[i];
    if (status[j].ok()) {
      ptr = (MetaData *)&(result[j][0]);
      memcpy(&(data[offset]), ptr->data, sizeof(Float) * this->dim_);
      global_step = std::max(global_step, ptr->update_logic_time);
    } else {
      //需要初始化
      key = mask_group(this->group_, keys[i]);
      std::string *tmpValue = this->create(key);
      s_put_keys.emplace_back(rocksdb::Slice((char *)&key, sizeof(u_int64_t)));
      put_result.emplace_back(*tmpValue);
      ptr = (MetaData *)&((*tmpValue)[0]);
      memcpy(&(data[offset]), ptr->data, sizeof(Float) * this->dim_);
      delete tmpValue;
    }
    offset += this->dim_;
  }

  assert(offset == n);

  //写到rocksdb里面去
  rocksdb::WriteOptions put_options;
  put_options.sync = false;
  rocksdb::WriteBatch batch;
  for (size_t i = 0; i < s_put_keys.size(); i++) {
    batch.Put(s_put_keys[i], put_result[i]);
  }
  db_->Write(put_options, &batch);
  return global_step;
}

void Embedding::apply_gradients(u_int64_t *keys, int len, Float *gds, int n,
                                u_int64_t global_step) {
  //写入的时候需要加锁,将key按照锁进行分组
  std::vector<rocksdb::Slice> s_keys_per_lock[max_lock_num]; //分组的key
  std::vector<Float *> gds_per_lock[max_lock_num];           //分组的gd
  int locker_id;
  u_int64_t key;
  for (int i = 0; i < len; i++) {
    //能够push进去的,必须要filter check ok
    key = mask_group(this->group_, keys[i]);
    if (filter_ == nullptr || filter_->check(key)) {
      locker_id = key & (max_lock_num - 1);
      gds_per_lock[locker_id].emplace_back(&(gds[i * dim_]));
      s_keys_per_lock[locker_id].emplace_back(
          rocksdb::Slice((char *)&key, sizeof(u_int64_t)));
    } else {
      filter_->add(key);
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
    auto status = db_->MultiGet(get_options, s_keys_per_lock[i], &result);
    //遍历所有,更新值
    for (size_t j = 0; j < s_keys_per_lock[i].size(); j++) {
      if (status[j].ok()) {
        //更新数据
        this->update((MetaData *)&(result[j][0]), gds_per_lock[i][j],
                     global_step);
        batch.Put(s_keys_per_lock[i][j], result[i]);
      }
    }
    //写回到rocksdb
    db_->Write(put_options, &batch);
    lockers_[i].unlock();
    s_keys_per_lock[i].clear();
    gds_per_lock[i].clear();
  }
}

void Embedding::dump(std::string path, int expires) {
  //新建快照
  rocksdb::ReadOptions read_option;
  const rocksdb::Snapshot *sp = this->db_->GetSnapshot();
  read_option.snapshot = sp;
  rocksdb::Iterator *it = this->db_->NewIterator(rocksdb::ReadOptions());
  MetaData *ptr;
  auto oldest_timestamp = get_current_time() - 86400 * expires;

  //把数据放到cache中去
  std::vector<char *> cache;
  char *data;
  int all_length = sizeof(u_int64_t) + this->dim_ * sizeof(Float);
  int data_length = this->dim_ * sizeof(Float);

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    ptr = (MetaData *)it->value().data();
    if (ptr->update_real_time < oldest_timestamp) {
      continue;
    }
    data = (char *)malloc(all_length);
    memcpy(data, &(ptr->key), sizeof(u_int64_t));
    memcpy(data + sizeof(u_int64_t), ptr->data, data_length);
    cache.push_back(data);
  }
  this->db_->ReleaseSnapshot(sp);

  //大小比较
  auto cmp = [](const char *a, const char *b) -> int {
    if (((u_int64_t *)a)[0] > ((u_int64_t *)b)[0]) {
      return 1;
    } else if (((u_int64_t *)a)[0] == ((u_int64_t *)b)[0]) {
      return 0;
    } else {
      return -1;
    }
  };

  //从小到大排序
  std::sort(cache.begin(), cache.end(), cmp);
  size_t size = cache.size();
  std::ofstream writer(path, std::ios::out | std::ios::binary);
  //写值  int(dim), size_t(size),data
  writer.write((char *)(&this->dim_), sizeof(int));
  writer.write((char *)&size, sizeof(size_t));
  for (size_t i = 0; i < size; i++) {
    writer.write(cache[i], all_length);
    free(cache[i]);
    cache[i] = nullptr;
  }
  writer.close();
  cache.clear();
}