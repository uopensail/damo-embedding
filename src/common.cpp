#include "common.h"

Params::Params() {}

Params::Params(json &params) : params_(params) {}

Params::Params(const std::string &str) { params_ = json::parse(str); }

Params::Params(const Params &p) : params_(p.params_) {}

const bool Params::isnil() const { return this->params_.size() == 0; }

Params &Params::operator=(const Params &p) {
  if (this == &p) {
    return *this;
  }
  params_ = p.params_;
  return *this;
}

std::string Params::to_json() const { return this->params_.dump(); }

Params::~Params() {}

bool Params::contains(const std::string &key) {
  return this->params_.contains(key);
}

int64_t get_current_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return static_cast<int64_t>(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

Float safe_sqrt(Float x) { return x >= 0.0 ? sqrtf((x)) : 0.0; }

Float sign(Float x) { return x >= 0.0 ? 1.0 : -1.0; }
