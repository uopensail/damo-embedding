#include "initializer.h"

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>

namespace embedding {

// RandomUniform initializer implementation ===========================
RandomUniform::RandomUniform(const Params& initializer_params)
    : Initializer(initializer_params), random_(std::random_device{}()) {}

std::string RandomUniform::to_string() const { return "random_uniform"; }

void RandomUniform::call(float* data, int dim) {
  if (dim <= 0) {
    throw std::invalid_argument("Dimension must be positive");
  }
  if (!data) {
    throw std::invalid_argument("Data pointer is null");
  }

  const float scale = std::sqrt(1.0f / static_cast<float>(dim));
  std::uniform_real_distribution<float> scaled_distribution(-scale, scale);

  for (int i = 0; i < dim; ++i) {
    data[i] = scaled_distribution(random_);
  }
}

// RandomNormal initializer implementation ============================
RandomNormal::RandomNormal(const Params& initializer_params)
    : Initializer(initializer_params),
      mean_(initializer_params.get<float>("mean", 0.0)),
      stddev_(initializer_params.get<float>("stddev", 1.0)),
      distribution_(mean_, stddev_),
      random_(std::random_device{}()) {
  if (stddev_ <= 0.0) {
    throw std::invalid_argument("Standard deviation must be positive");
  }
}

std::string RandomNormal::to_string() const {
  return "random_normal: (mean=" + std::to_string(mean_) +
         ", std=" + std::to_string(stddev_) + ")";
}

void RandomNormal::call(float* data, int dim) {
  if (dim <= 0) {
    throw std::invalid_argument("Dimension must be positive");
  }
  if (!data) {
    throw std::invalid_argument("Data pointer is null");
  }

  for (int i = 0; i < dim; ++i) {
    data[i] = distribution_(random_);
  }
}

// XavierUniform initializer implementation ===========================
XavierUniform::XavierUniform(const Params& initializer_params)
    : Initializer(initializer_params), random_(std::random_device{}()) {}

void XavierUniform::call(float* data, int dim) {
  if (dim <= 0) {
    throw std::invalid_argument("Dimension must be positive");
  }
  if (!data) {
    throw std::invalid_argument("Data pointer is null");
  }

  const float scale = std::sqrt(6.0f / static_cast<float>(dim));
  std::uniform_real_distribution<float> scaled_distribution(-scale, scale);

  for (int i = 0; i < dim; ++i) {
    data[i] = scaled_distribution(random_);
  }
}

std::string XavierUniform::to_string() const { return "xavier_uniform"; }

// Factory function implementation ====================================
std::shared_ptr<Initializer> get_initializers(
    const Params& initializer_params) {
  const auto name = initializer_params.get<std::string>("name", "zeros");

  if (name == "xavier") {
    return std::make_shared<XavierUniform>(initializer_params);
  } else if (name == "uniform") {
    return std::make_shared<RandomUniform>(initializer_params);
  } else if (name == "random") {
    return std::make_shared<RandomNormal>(initializer_params);
  }

  throw std::invalid_argument("Unsupported initializer type: " + name);
}

}  // namespace embedding
