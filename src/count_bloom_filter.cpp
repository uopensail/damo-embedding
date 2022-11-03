#include "count_bloom_filter.h"

void flush_thread_func(CountBloomFilter *filter) {
  for (; CountBloomFilterGlobalStatus.load();) {
    std::this_thread::sleep_for(std::chrono::seconds(120));
    filter->dump();
  }
}

CountBloomFilter::CountBloomFilter(size_t capacity, int count,
                                   std::string filename, bool reload,
                                   double ffp)
    : ffp_(ffp), capacity_(capacity), filename_(filename), count_(count) {
  //计算需要的空间: -(n*ln(p))/ (ln2)^2
  this->size_ =
      size_t(log(1.0 / ffp_) * double(capacity_) / (log(2.0) * log(2.0)));
  if (this->size_ & 1) {
    this->size_++;
  }
  auto half = this->size_ >> 1;
  //计算hash函数的个数：k=ln(2)*m/n
  this->k_ = int(log(2.0) * double(this->size_) / double(this->capacity_));

  bool need_create_file = true;
  if (reload) {
    if (access(filename.c_str(), 0) == 0) {
      struct stat info;
      stat(filename.c_str(), &info);
      //判定是否符合条件, 是否要创建文件
      if (size_t(info.st_size) == half * sizeof(BiCounter)) {
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
    fseek(w, half * sizeof(BiCounter) - 1, SEEK_SET);
    fwrite(&tmp, 1, 1, w);
    fclose(w);
  }

  this->fp_ = open(filename.c_str(), O_RDWR, 0777);
  this->data_ = (BiCounter *)mmap(0, half * sizeof(BiCounter),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, fp_, 0);
  if (this->data_ == MAP_FAILED) {
    exit(-1);
  }

  if (need_create_file) {
    memset(this->data_, 0, half * sizeof(BiCounter));
  }

  this->flush_thread_ = std::thread(flush_thread_func, this);
  this->handler_ = flush_thread_.native_handle();
  this->flush_thread_.detach();
}

void CountBloomFilter::dump() {
  auto half = this->size_ >> 1;
  msync((void *)this->data_, sizeof(BiCounter) * half, MS_ASYNC);
}

//检查在不在，次数是否大于count
bool CountBloomFilter::check(const u_int64_t &key) {
  int min_count = MaxCount;
  u_int64_t hash = key;
  for (int i = 0; i < this->k_; i++) {
    auto idx = hash % this->size_;
    if (idx & 1) {
      idx >>= 1;
      min_count =
          this->data_[idx].m2 < min_count ? this->data_[idx].m2 : min_count;
    } else {
      idx >>= 1;
      min_count =
          this->data_[idx].m1 < min_count ? this->data_[idx].m1 : min_count;
    }
    hash = hash_func(hash);
  }
  return min_count >= count_;
}

void CountBloomFilter::add(const u_int64_t &key, u_int64_t num) {
  u_int64_t hash = key;
  for (int i = 0; i < this->k_; i++) {
    auto idx = hash % this->size_;
    if (idx & 1) {
      idx >>= 1;
      this->data_[idx].m2 = this->data_[idx].m2 + num < MaxCount
                                ? this->data_[idx].m2 + num
                                : MaxCount;
    } else {
      idx >>= 1;
      this->data_[idx].m1 = this->data_[idx].m1 + num < MaxCount
                                ? this->data_[idx].m1 + num
                                : MaxCount;
    }
    hash = hash_func(hash);
  }
}

CountBloomFilter::~CountBloomFilter() {
  CountBloomFilterGlobalStatus.store(false);
  this->dump();
  auto half = this->size_ >> 1;
  munmap((void *)this->data_, sizeof(BiCounter) * half);
  close(this->fp_);
  pthread_cancel(this->handler_);
}
