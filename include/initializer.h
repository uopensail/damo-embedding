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

#ifndef DAMO_EMBEDDING_INITIALIZER_H_
#define DAMO_EMBEDDING_INITIALIZER_H_

#pragma once

#include <iostream>
#include <memory>
#include <random>

#include "common.h"

namespace embedding {

/**
 * @brief Abstract base class for weight initializers.
 */
class Initializer {
 public:
  // Disable default constructor and copy operations
  Initializer() = delete;
  Initializer(const Initializer&) = delete;
  Initializer& operator=(const Initializer&) = delete;

  explicit Initializer(const Params& initializer_params)
      : name_(initializer_params.get<std::string>("name")) {}
  virtual ~Initializer() = default;

  /**
   * @brief Get the name of the initializer.
   * @return The name of the initializer.
   */
  const std::string& get_name() const { return name_; }

  /**
   * @brief Convert initializer to string representation.
   * @return String representation of the initializer.
   */
  virtual std::string to_string() const = 0;

  /**
   * @brief Initialize the weights.
   * @param data Pointer to the data array to be initialized.
   * @param dim Dimension of the data array.
   */
  virtual void call(float* data, int dim) = 0;

 protected:
  std::string name_;
};

/**
 * @brief Uniform random weight initializer.
 */
class RandomUniform : public Initializer {
 public:
  // Disable default constructor and copy operations
  RandomUniform() = delete;
  RandomUniform(const RandomUniform&) = delete;
  RandomUniform& operator=(const RandomUniform&) = delete;

  explicit RandomUniform(const Params& initializer_params);
  ~RandomUniform() override = default;

  void call(float* data, int dim) override;
  std::string to_string() const override;

 private:
  std::default_random_engine random_;
};

/**
 * @brief Normal random weight initializer.
 */
class RandomNormal : public Initializer {
 public:
  // Disable default constructor and copy operations
  RandomNormal() = delete;
  RandomNormal(const RandomNormal&) = delete;
  RandomNormal& operator=(const RandomNormal&) = delete;

  explicit RandomNormal(const Params& initializer_params);
  ~RandomNormal() override = default;

  void call(float* data, int dim) override;
  std::string to_string() const override;

 private:
  double mean_;
  double stddev_;
  std::normal_distribution<float> distribution_;
  std::default_random_engine random_;
};

/**
 * @brief Xavier uniform weight initializer.
 */
class XavierUniform : public Initializer {
 public:
  // Disable default constructor and copy operations
  XavierUniform() = delete;
  XavierUniform(const XavierUniform&) = delete;
  XavierUniform& operator=(const XavierUniform&) = delete;

  explicit XavierUniform(const Params& initializer_params);
  ~XavierUniform() override = default;

  void call(float* data, int dim) override;
  std::string to_string() const override;

 private:
  std::default_random_engine random_;
};

/**
 * @brief Factory function to create an initializer based on parameters.
 * @param p The parameters for creating the initializer.
 * @return A shared pointer to the created initializer.
 */
std::shared_ptr<Initializer> get_initializers(const Params& p);

}  // namespace embedding

#endif  // DAMO_EMBEDDING_INITIALIZER_H_