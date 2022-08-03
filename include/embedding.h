#ifndef DAMO_EMBEDDING_EMBEDDING_H
#define DAMO_EMBEDDING_EMBEDDING_H

#pragma once

#include "count_bloom_filter.h"
#include "initializer.h"
#include "optimizer.h"
#include <mutex>
#include <rocksdb/db.h>
#include <rocksdb/write_batch.h>

//最大的锁的个数
#define max_lock_num 1024

class Embedding
{
private:
    rocksdb::DB *db_;
    int dim_;
    u_int64_t lag_;
    std::shared_ptr<Optimizer> optimizer_;
    std::shared_ptr<Initializer> initializer_;
    std::shared_ptr<CountBloomFilter> filter_;
    std::mutex lockers_[max_lock_num];

private:
    void create(u_int64_t &key, std::string &value);
    void update(MetaData *ptr, Float *gds, u_int64_t global_step);

public:
    Embedding(int dim, u_int64_t step_lag, std::string data_dir, std::shared_ptr<Optimizer> optimizer, std::shared_ptr<Initializer> initializer,
              std::shared_ptr<CountBloomFilter> filter);
    ~Embedding();
    u_int64_t lookup(u_int64_t *keys, int len, Float *data, int n);
    void apply_gradients(u_int64_t *keys, int len, Float *gds, int n, u_int64_t global_steps);
    void dump(std::string path, int expires);
};

#endif // DAMO_EMBEDDING_EMBEDDING_H