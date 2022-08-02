#ifndef DAMO_EMBEDDING_SPARSE_PSERVER_H
#define DAMO_EMBEDDING_SPARSE_PSERVER_H
#pragma once

//稀疏的参数服务器
class SparsePServer
{
private:
    std::shared_ptr<RocksDBStorage> storage;
    std::shared_ptr<SparseConfig> global_config;
    std::shared_ptr<CountBloomFilter> count_bloom_filter;

private:
    void init(u_int64_t &key, std::string &value);

public:
    SparsePServer() = delete;

    SparsePServer(const SparsePServer &) = delete;

    SparsePServer(const SparsePServer &&) = delete;

    SparsePServer(std::string config_path);

    //批量pull
    void mpull(u_int64_t *keys, int len, Float *data, int n);

    void pull(u_int64_t &key, Float *data, int n);

    bool push(u_int64_t &key, Float *grad, int n, long long global_step);

    //批量push
    void mpush(u_int64_t *keys, int len, Float *gds, int n, long long global_step);

    int &get_dim(u_int64_t &key);

    void dump(std::string &filename, int days);

    ~SparsePServer();
};

#endif // DAMO_EMBEDDING_SPARSE_PSERVER_H