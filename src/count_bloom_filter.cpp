#include "count_bloom_filter.h"

void flush_thread_func(CountBloomFilter *filter)
{
    for (; CountBloomFilterGlobalStatus.load();)
    {
        std::this_thread::sleep_for(std::chrono::seconds(120));
        filter->dump();
    }
}

CountBloomFilter::CountBloomFilter(size_t capacity, int count, std::string filename, bool reload,
                                   double ffp)
    : ffp_(ffp), capacity_(capacity), filename_(filename), count_(count)
{
    //计算需要的空间：-(n*ln(p))/ (ln2)^2
    size_ = size_t(log(1.0 / ffp_) * double(capacity_) / (log(2.0) * log(2.0)));
    if (size_ & 1)
    {
        size_++;
    }
    auto half = size_ >> 1;
    //计算hash函数的个数：k=ln(2)*m/n
    k = int(log(2.0) * double(size_) / double(capacity_));

    bool need_create_file = true;
    if (reload)
    {
        if (access(filename.c_str(), 0) == 0)
        {
            struct stat info;
            stat(filename.c_str(), &info);
            if (size_t(info.st_size) == half * sizeof(BiCounter))
            {
                need_create_file = false;
            }
            else
            {
                remove(filename.c_str());
            }
        }
    }

    //创建文件
    if (need_create_file)
    {
        FILE *w = fopen(filename.c_str(), "wb");
        char tmp = '\0';
        fseek(w, half * sizeof(BiCounter) - 1, SEEK_SET);
        fwrite(&tmp, 1, 1, w);
        fclose(w);
    }

    fp_ = open(filename.c_str(), O_RDWR, 0777);
    data_ = (BiCounter *)mmap(0, half * sizeof(BiCounter), PROT_READ | PROT_WRITE, MAP_SHARED, fp_, 0);
    if (data_ == MAP_FAILED)
    {
        exit(-1);
    }

    if (need_create_file)
    {
        memset(data_, 0, half * sizeof(BiCounter));
    }

    flush_thread_ = std::thread(flush_thread_func, this);
    handler_ = flush_thread_.native_handle();
    flush_thread_.detach();
}

void CountBloomFilter::dump()
{
    auto half = size_ >> 1;
    msync((void *)data_, sizeof(BiCounter) * half, MS_ASYNC);
}

//检查在不在，次数是否大于count
bool CountBloomFilter::check(const u_int64_t &key)
{
    int min_count = MaxCount;
    u_int64_t hash = key;
    for (int i = 0; i < k; i++)
    {
        auto idx = hash % size_;
        if (idx & 1)
        {
            idx >>= 1;
            min_count = data_[idx].m2 < min_count ? data_[idx].m2 : min_count;
        }
        else
        {
            idx >>= 1;
            min_count = data_[idx].m1 < min_count ? data_[idx].m1 : min_count;
        }
        hash = hash_func(hash);
    }
    return min_count >= count_;
}

void CountBloomFilter::add(const u_int64_t &key, u_int64_t num)
{
    u_int64_t hash = key;
    for (int i = 0; i < k; i++)
    {
        auto idx = hash % size_;
        if (idx & 1)
        {
            idx >>= 1;
            data_[idx].m2 = data_[idx].m2 + num < MaxCount ? data_[idx].m2 + num : MaxCount;
        }
        else
        {
            idx >>= 1;
            data_[idx].m1 = data_[idx].m1 + num < MaxCount ? data_[idx].m1 + num : MaxCount;
        }
        hash = hash_func(hash);
    }
}

CountBloomFilter::~CountBloomFilter()
{
    CountBloomFilterGlobalStatus.store(false);
    dump();
    auto half = size_ >> 1;
    munmap((void *)data_, sizeof(BiCounter) * half);
    close(fp_);
    pthread_cancel(handler_);
}
