#include "common.h"

Params::Params(const std::shared_ptr<cpptoml::table> &table) : table(table) {}

Params::Params(const Params &p) : table(p.table) {}

const bool Params::isnil() const { return this->table == nullptr; }

Params &Params::operator=(const Params &p) {
  if (this == &p) {
    return *this;
  }
  table = p.table;
  return *this;
}

Params &Params::operator=(const std::shared_ptr<cpptoml::table> &table) {
  this->table = table;
  return *this;
}

Params::~Params() {}

bool Params::contains(const std::string &key) {
  return this->table->contains(key);
}

int64_t get_current_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return static_cast<int64_t>(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

Float safe_sqrt(Float x) { return x >= 0.0 ? sqrtf((x)) : 0.0; }

Float sign(Float x) { return x >= 0.0 ? 1.0 : -1.0; }
