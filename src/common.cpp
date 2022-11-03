#include "common.h"

Params::Params(const std::shared_ptr<cpptoml::table> &table) : table(table) {}

Params::Params(const Params &p) : table(p.table) {}

Params &Params::operator=(const Params &p) {
  if (this == &p) {
    return *this;
  }
  table = p.table;
  return *this;
}

Params::~Params() {}
