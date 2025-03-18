#include "optimizer.h"

#include <cmath>      // For math functions
#include <memory>     // For smart pointers
#include <stdexcept>  // For standard exceptions

namespace embedding {

// AdamwOptimizer implementation =====================================

AdamWOptimizer::AdamWOptimizer(const Params& optimizer_params)
    : Optimizer(optimizer_params),
      gamma_(optimizer_params.get<float>("gamma", 0.001f)),
      beta1_(optimizer_params.get<float>("beta1", 0.9f)),
      beta2_(optimizer_params.get<float>("beta2", 0.999f)),
      lambda_(optimizer_params.get<float>("lambda", 0.01f)),
      epsilon_(optimizer_params.get<float>("epsilon", 1e-8f)) {
  // Validate hyperparameters
  if (gamma_ <= 0.0f) {
    throw std::invalid_argument("Learning rate (gamma) must be positive");
  }
  if (beta1_ <= 0.0f || beta1_ >= 1.0f) {
    throw std::invalid_argument("Beta1 must be in (0, 1) range");
  }
  if (beta2_ <= 0.0f || beta2_ >= 1.0f) {
    throw std::invalid_argument("Beta2 must be in (0, 1) range");
  }
  if (lambda_ < 0.0f) {
    throw std::invalid_argument("Weight decay (lambda) cannot be negative");
  }
}

std::string AdamWOptimizer::to_string() const {
  return "{\"AdamW\": { \"gamma\":" + std::to_string(gamma_) +
         ", \"beta1\":" + std::to_string(beta1_) +
         ", \"beta2\":" + std::to_string(beta2_) +
         ", \"lambda\":" + std::to_string(lambda_) +
         ", \"epsilon\":" + std::to_string(epsilon_) + "}}";
}

int AdamWOptimizer::get_space(int dim) const {
  if (dim <= 0) {
    throw std::invalid_argument("Dimension must be positive");
  }
  return 3 * dim;  // Space for weights, first and second moment estimates
}

void AdamWOptimizer::call(float* data, const float* grads, int dim,
                          int64_t global_step) {
  if (!data || !grads) {
    throw std::invalid_argument("Invalid data or gradient pointers");
  }
  if (dim <= 0) {
    throw std::invalid_argument("Dimension must be positive");
  }
  if (global_step < 0) {
    throw std::invalid_argument("Global step must be non-negative");
  }

  float* weights = data;
  float* m = data + dim;      // First moment estimate
  float* v = data + 2 * dim;  // Second moment estimate

  const float beta1_pow = std::pow(beta1_, static_cast<float>(global_step));
  const float beta2_pow = std::pow(beta2_, static_cast<float>(global_step));
  const float lr = gamma_;  // Learning rate

  for (int i = 0; i < dim; ++i) {
    // Apply weight decay
    weights[i] -= lambda_ * lr * weights[i];

    // Update first and second moment estimates
    m[i] = beta1_ * m[i] + (1.0f - beta1_) * grads[i];
    v[i] = beta2_ * v[i] + (1.0f - beta2_) * grads[i] * grads[i];

    // Compute bias-corrected estimates
    const float m_hat = m[i] / (1.0f - beta1_pow);
    const float v_hat = v[i] / (1.0f - beta2_pow);

    // Update weights
    weights[i] -= lr * m_hat / (safe_sqrt(v_hat) + epsilon_);
  }
}

// Optimizer factory function =========================================

std::shared_ptr<Optimizer> get_optimizers(const Params& optimizer_params) {
  const auto name = optimizer_params.get<std::string>("name", "adamw");

  if (name == "adamw") {
    return std::make_shared<AdamWOptimizer>(optimizer_params);
  }

  throw std::invalid_argument("Unsupported optimizer type: " + name);
}

}  // namespace embedding
