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

#ifndef DAMO_EMBEDDING_OPTIMIZER_H
#define DAMO_EMBEDDING_OPTIMIZER_H

#pragma once

#include <iostream>

#include "common.h"
#include "initializer.h"
#include "learning_rate_scheduler.h"

class Optimizer {
 public:
  Optimizer() = delete;
  Optimizer(const Optimizer &) = delete;
  Optimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~Optimizer();
  const std::string &get_name();

  /**
   * @brief get the space of the optimizer
   *
   * @param dim weight dimension
   * @return int
   */
  virtual int get_space(int dim);

  /**
   * @brief according to the global step and original learning rate, to generate
   * lr
   *
   * @param global_step global step
   * @param learning_rate original learning rate
   * @return Float
   */
  inline Float get_lr(int64_t global_step, Float learning_rate);

  /**
   * @brief call the optimizer, updating the embedding
   *
   * @param w weights
   * @param wn width of the weights
   * @param gds gradients for weights
   * @param gn width of the grad
   * @param global_step global step
   */
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step) = 0;

 protected:
  std::string name_;
  lr_scheduler function_;
  Params scheduler_;
};

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer() = delete;
  SGDOptimizer(const SGDOptimizer &) = delete;
  SGDOptimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~SGDOptimizer();
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step);

 private:
  Float gamma_;
  Float lambda_;
};

class FTRLOptimizer : public Optimizer {
 public:
  FTRLOptimizer() = delete;
  FTRLOptimizer(const FTRLOptimizer &) = delete;
  FTRLOptimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~FTRLOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step);

 private:
  Float alpha_;
  Float beta_;
  Float lambda1_;
  Float lambda2_;
};

class AdagradOptimizer : public Optimizer {
 public:
  AdagradOptimizer() = delete;
  AdagradOptimizer(const AdagradOptimizer &) = delete;
  AdagradOptimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~AdagradOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step);

 private:
  Float gamma_;
  Float lambda_;
  Float eta_;
  Float epsilon_;
};

/**
 * @brief this is Adam with L2 regularization
 *
 */
class AdamOptimizer : public Optimizer {
 public:
  AdamOptimizer() = delete;
  AdamOptimizer(const AdamOptimizer &) = delete;
  AdamOptimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~AdamOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step);

 private:
  Float gamma_;
  Float beta1_;
  Float beta2_;
  Float lambda_;
  Float epsilon_;
};

class AmsGradOptimizer : public Optimizer {
 public:
  AmsGradOptimizer() = delete;
  AmsGradOptimizer(const AmsGradOptimizer &) = delete;
  AmsGradOptimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~AmsGradOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step);

 private:
  Float gamma_;
  Float beta1_;
  Float beta2_;
  Float lambda_;
  Float epsilon_;
};

class AdamWOptimizer : public Optimizer {
 public:
  AdamWOptimizer() = delete;
  AdamWOptimizer(const AdamWOptimizer &) = delete;
  AdamWOptimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~AdamWOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step);

 private:
  Float gamma_;
  Float beta1_;
  Float beta2_;
  Float lambda_;
  Float epsilon_;
};

class LionOptimizer : public Optimizer {
 public:
  LionOptimizer() = delete;
  LionOptimizer(const LionOptimizer &) = delete;
  LionOptimizer(const Params &optimizer_params, const Params &scheduler);
  virtual ~LionOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, int64_t global_step);

 private:
  Float eta_;
  Float beta1_;
  Float beta2_;
  Float lambda_;
};

std::shared_ptr<Optimizer> get_optimizers(const Params &optimizer_params,
                                          const Params &scheduler);

#endif  // DAMO_EMBEDDING_OPTIMIZER_H
