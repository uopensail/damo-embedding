#include "counting_bloom_filter.h"

void flush_thread_func(CountingBloomFilter *filter) {
  for (; CountBloomFilterGlobalStatus.load();) {
    std::this_thread::sleep_for(std::chrono::seconds(120));
    filter->dump();
  }
}

CountingBloomFilter::CountingBloomFilter(const Params &config)
    : CountingBloomFilter(
          config.get<u_int64_t>("capacity"), config.get<int>("count"),
          config.get<std::string>("path"), config.get<bool>("reload")) {}

CountingBloomFilter::CountingBloomFilter(size_t capacity, int count,
                                         std::string filename, bool reload,
                                         double ffp)
    : ffp_(ffp), capacity_(capacity), filename_(filename), count_(count) {
  //计算需要的空间: -(n*ln(p))/ (ln2)^2
  this->size_ =
      size_t(log(1.0 / ffp_) * double(capacity_) / (log(2.0) * log(2.0)));
  if (this->size_ & 1) {
    this->size_++;
  }

  //计算hash函数的个数：k=ln(2)*m/n
  this->k_ = int(log(2.0) * double(this->size_) / double(this->capacity_));

  bool need_create_file = true;
  if (reload) {
    if (access(filename.c_str(), 0) == 0) {
      struct stat info;
      stat(filename.c_str(), &info);
      //判定是否符合条件, 是否要创建文件
      if (size_t(info.st_size) == this->size_ * sizeof(Counter)) {
        need_create_file = false;
      } else {
        remove(filename.c_str());
      }
    }
  }

  //创建文件
  if (need_create_file) {
    FILE *w = fopen(filename.c_str(), "wb");
    char tmp = '\0';
    fseek(w, this->size_ * sizeof(Counter) - 1, SEEK_SET);
    fwrite(&tmp, 1, 1, w);
    fclose(w);
  }

  this->fp_ = open(filename.c_str(), O_RDWR, 0777);
  this->data_ = (Counter *)mmap(0, this->size_ * sizeof(Counter),
                                PROT_READ | PROT_WRITE, MAP_SHARED, fp_, 0);
  if (this->data_ == MAP_FAILED) {
    exit(-1);
  }

  if (need_create_file) {
    memset(this->data_, 0, this->size_ * sizeof(Counter));
  }

  this->flush_thread_ = std::thread(flush_thread_func, this);
  this->handler_ = flush_thread_.native_handle();
  this->flush_thread_.detach();
}

void CountingBloomFilter::dump() {
  msync((void *)this->data_, this->size_ * sizeof(Counter), MS_ASYNC);
}

//检查在不在，次数是否大于count
bool CountingBloomFilter::check(const u_int64_t &key) {
  int min_count = MaxCount;
  u_int64_t hash = key;
  unsigned char *value;
  for (int i = 0; i < this->k_; i++) {
    auto idx = hash % this->size_;
    value = (unsigned char *)&(this->data_[idx]);
    min_count = *value < min_count ? *value : min_count;
    hash = hash_func(hash);
  }
  return min_count >= count_;
}

void CountingBloomFilter::add(const u_int64_t &key, u_int64_t num) {
  u_int64_t hash = key;
  unsigned char *value;
  int tmp;
  for (int i = 0; i < this->k_; i++) {
    auto idx = hash % this->size_;
    value = (unsigned char *)&(this->data_[idx]);
    tmp = *value + num;
    *value = tmp < MaxCount ? tmp : MaxCount;
    hash = hash_func(hash);
  }
}

CountingBloomFilter::~CountingBloomFilter() {
  CountBloomFilterGlobalStatus.store(false);
  this->dump();
  munmap((void *)this->data_, sizeof(Counter) * this->size_);
  close(this->fp_);
  pthread_cancel(this->handler_);
}
