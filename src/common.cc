#include "common.h"

#include <sys/time.h>  // For system time functions

#include <cerrno>     // For errno
#include <cstring>    // For strerror
#include <stdexcept>  // For standard exceptions

namespace embedding {

// Constructor from JSON object
Params::Params(const nlohmann::json& params) : params_(params) {}

// Constructor from JSON string
Params::Params(const std::string& str) {
  try {
    params_ = nlohmann::json::parse(str);
  } catch (const nlohmann::json::parse_error& e) {
    throw std::invalid_argument("JSON parse error: " + std::string(e.what()));
  }
}

// Get the current time in milliseconds
int64_t get_current_time() {
  struct timeval tv;
  if (gettimeofday(&tv, nullptr) != 0) {
    throw std::runtime_error("Failed to get current time: " +
                             std::string(strerror(errno)));
  }
  return static_cast<int64_t>(tv.tv_sec) * 1000 +
         static_cast<int64_t>(tv.tv_usec) / 1000;
}

// Safely calculate square root
float safe_sqrt(float x) {
  if (x < 0.0f) {
    throw std::domain_error("Cannot take square root of negative value");
  }
  return std::sqrt(x);
}

// Sign function
float sign(float x) { return (x >= 0.0f) ? 1.0f : -1.0f; }

}  // namespace embedding
