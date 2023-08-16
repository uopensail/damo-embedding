#include "counting_bloom_filter.h"

uint64_t hash_func(const int64_t &x) {
  return ((x >> 31) & high_mask) | ((x & low_mask) << 33);
}

uint64_t hash_func(const Key &x) {
  uint64_t hash = x.group;
  return hash_func(x.key) | (hash << 15);
}

void create_empty_file(const std::string &filename, const size_t &size) {
  FILE *w = fopen(filename.c_str(), "wb");
  char tmp = '\0';
  fseek(w, size - 1, SEEK_SET);
  fwrite(&tmp, 1, 1, w);
  fclose(w);
}

CountingBloomFilter::CountingBloomFilter()
    : CountingBloomFilter(min_size, max_count,
                          "/tmp/COUNTING_BLOOM_FILTER_DATA", true, FPR) {}

CountingBloomFilter::CountingBloomFilter(const Params &config)
    : CountingBloomFilter(
          config.get<uint64_t>("capacity", min_size),
          config.get<int>("count", max_count),
          config.get<std::string>("path", "/tmp/COUNTING_BLOOM_FILTER_DATA"),
          config.get<bool>("reload", true), config.get<double>("fpr", FPR)) {}

CountingBloomFilter::CountingBloomFilter(size_t capacity, int count,
                                         const std::string &filename,
                                         bool reload, double fpr)
    : fpr_(fpr), capacity_(capacity), filename_(filename), count_(count) {
  if (count > max_count) {
    std::cout << "counting bloom filter support max count is: " << max_count
              << std::endl;
    count_ = max_count;
  }
  // counter_num_ = -(n*ln(p))/ (ln2)^2
  this->counter_num_ =
      size_t(log(1.0 / fpr_) * double(capacity_) / (log(2.0) * log(2.0)));
  if (this->counter_num_ & 1) {
    this->counter_num_++;
  }

  // a unit space is half char
  this->space_ = this->counter_num_ >> 1;

  // k_=ln(2)*m/n
  this->k_ = int(
      ceil(log(2.0) * double(this->counter_num_) / double(this->capacity_)));

  bool need_create_file = true;
  if (reload) {
    if (access(filename.c_str(), 0) == 0) {
      struct stat info;
      stat(filename.c_str(), &info);
      if (size_t(info.st_size) == this->space_) {
        need_create_file = false;
      } else {
        remove(filename.c_str());
      }
    }
  }

  if (need_create_file) {
    create_empty_file(filename, this->space_);
  }

  this->fd_ = open(filename.c_str(), O_RDWR, 0777);
  this->data_ = (Counter *)mmap(0, this->space_, PROT_READ | PROT_WRITE,
                                MAP_SHARED, this->fd_, 0);

  if (this->data_ == MAP_FAILED) {
    exit(-1);
  }

  if (need_create_file) {
    memset(this->data_, 0, this->space_);
  }
}

bool CountingBloomFilter::check(const Key &key) {
  int min_count = max_count;
  uint64_t hash = hash_func(key);

  for (int i = 0; i < this->k_; i++) {
    auto idx = hash % this->counter_num_;
    if (idx & 1) {
      idx >>= 1;
      if (data_[idx].m2 < min_count) {
        min_count = data_[idx].m2;
      }
    } else {
      idx >>= 1;
      if (data_[idx].m1 < min_count) {
        min_count = data_[idx].m1;
      }
    }
    hash = hash_func(hash);
  }
  return min_count >= count_;
}

void CountingBloomFilter::add(const Key &key, const int64_t &num) {
  uint64_t hash = hash_func(key);
  for (int i = 0; i < this->k_; i++) {
    auto idx = hash % this->counter_num_;
    if (idx & 1) {
      idx >>= 1;
      if (data_[idx].m2 < max_count) {
        data_[idx].m2++;
      }

    } else {
      idx >>= 1;
      if (data_[idx].m1 < max_count) {
        data_[idx].m1++;
      }
    }
    hash = hash_func(hash);
  }
}

int CountingBloomFilter::get_count() const { return this->count_; }
CountingBloomFilter::~CountingBloomFilter() {
  msync((void *)this->data_, this->space_, MS_ASYNC);
  munmap((void *)this->data_, this->space_);
  close(this->fd_);
}