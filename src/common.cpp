#include "common.h"

Params::Params(std::shared_ptr<cpptoml::table> &table) : table(table) {}

Params::Params(std::shared_ptr<cpptoml::table> &&table) : table(table) {}

Params::Params(const Params &p) : table(p.table) {}

Params::Params(const Params &&p) : table(p.table) {}

Params &Params::operator=(const Params &p)
{
    table = p.table;
    return *this;
}

Params::~Params() {}

u_int64_t get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return u_int64_t(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}
