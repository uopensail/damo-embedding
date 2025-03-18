//
// `Damo-Embedding` - 'c++ tool for sparse parameter server'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// `Damo-Embedding` is provided under: GNU Affero General Public License
// (AGPL3.0) https://www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//

#ifndef DAMO_EMBEDDING_OPTIMIZER_H_
#define DAMO_EMBEDDING_OPTIMIZER_H_

#pragma once

#include <memory>
#include <string>

#include "common.h"
#include "initializer.h"

namespace embedding {

/**
 * @brief Base class for optimization algorithms.
 */
class Optimizer {
 public:
  // Disable copy operations
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;

  /**
   * @brief Constructs an optimizer with specified parameters.
   * @param optimizer_params Configuration parameters for the optimizer.
   */
  explicit Optimizer(const Params& optimizer_params)
      : name_(optimizer_params.get<std::string>("name")) {}
  virtual ~Optimizer() = default;

  /**
   * @brief Gets the name of the optimizer.
   * @return Constant reference to the optimizer's name.
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Converts optimizer configuration to string representation.
   * @return String describing the optimizer configuration.
   */
  virtual std::string to_string() const = 0;

  /**
   * @brief Gets the required workspace size for the optimizer.
   * @param dim Dimension of the weight vector.
   * @return Required workspace size in elements.
   */
  virtual int get_space(int dim) const = 0;

  /**
   * @brief Updates parameters using optimization algorithm.
   * @param data Pointer to parameter data array.
   * @param grads Pointer to gradient data array.
   * @param dim Dimension of the parameter vector.
   * @param global_step Current global training step.
   */
  virtual void call(float* data, const float* grads, int dim,
                    int64_t global_step) = 0;

 protected:
  std::string name_;  ///< Name identifier for the optimizer
};

/**
 * @brief AdamW optimizer implementation with weight decay.
 */
class AdamWOptimizer : public Optimizer {
 public:
  // Disable copy operations
  AdamWOptimizer(const AdamWOptimizer&) = delete;
  AdamWOptimizer& operator=(const AdamWOptimizer&) = delete;

  /**
   * @brief Constructs AdamW optimizer with parameters.
   * @param optimizer_params Configuration parameters for AdamW.
   */
  explicit AdamWOptimizer(const Params& optimizer_params);

  /**
   * @brief Gets required workspace size for AdamW.
   * @param dim Parameter dimension.
   * @return Workspace size requirement (2 * dim for moments).
   */
  int get_space(int dim) const override;

  /**
   * @brief Performs AdamW optimization step.
   */
  void call(float* data, const float* grads, int dim,
            int64_t global_step) override;

  /**
   * @brief Returns string representation of AdamW configuration.
   */
  std::string to_string() const override;

 private:
  // Hyperparameters with default values
  float gamma_;    ///< Learning rate
  float beta1_;    ///< Exponential decay rate for 1st moment estimates
  float beta2_;    ///< Exponential decay rate for 2nd moment estimates
  float lambda_;   ///< Weight decay rate
  float epsilon_;  ///< Numerical stability term
};

/**
 * @brief Factory function to create optimizers.
 * @param optimizer_params Parameters specifying optimizer configuration.
 * @return Shared pointer to created optimizer instance.
 * @throws std::invalid_argument for unknown optimizer types.
 */
std::shared_ptr<Optimizer> get_optimizers(const Params& optimizer_params);

}  // namespace embedding

#endif  // DAMO_EMBEDDING_OPTIMIZER_H_