//
// `Damo-Embedding` - 'c++ tool for sparse parameter server'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
//
// This file is part of `Damo-Embedding`.
//
// `Damo-Embedding` is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// `Damo-Embedding` is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with `Damo-Embedding`.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef DAMO_EMBEDDING_OPTIMIZER_H
#define DAMO_EMBEDDING_OPTIMIZER_H

#pragma once

#include <iostream>

#include "common.h"
#include "decay_learning_rate.h"
#include "initializer.h"

class Optimizer {
 protected:
  std::string name_;
  decay_lr_func function_;
  Params decay_params_;

 public:
  Optimizer() = delete;
  Optimizer(const Optimizer &) = delete;
  Optimizer(const Params &optimizer_params, const Params &decay_params);
  virtual ~Optimizer();
  const std::string &get_name();
  virtual int get_space(int dim);
  inline Float get_decay_rate(u_int64_t global_step, Float learning_rate);
  virtual void call(Float *data, Float *gds, int dim,
                    u_int64_t global_step) = 0;
};

class SGDOptimizer : public Optimizer {
 private:
  Float eta;

 public:
  SGDOptimizer() = delete;
  SGDOptimizer(const SGDOptimizer &) = delete;
  SGDOptimizer(const Params &optimizer_params, const Params &decay_params);
  virtual ~SGDOptimizer();
  virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

class FTRLOptimizer : public Optimizer {
 private:
  Float alpha;
  Float beta;
  Float lambda1;
  Float lambda2;

 public:
  FTRLOptimizer() = delete;
  FTRLOptimizer(const FTRLOptimizer &) = delete;
  FTRLOptimizer(const Params &optimizer_params, const Params &decay_params);
  virtual ~FTRLOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

/**
 * @brief this is Adam with L2 regularization
 *
 */
class AdamOptimizer : public Optimizer {
 private:
  Float alpha;  // lr
  Float beta1;
  Float beta2;
  Float lambda;  // L2 regularization param
  Float epsilon;

 public:
  AdamOptimizer() = delete;
  AdamOptimizer(const AdamOptimizer &) = delete;
  AdamOptimizer(const Params &optimizer_params, const Params &decay_params);
  virtual ~AdamOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

class AmsGradOptimizer : public Optimizer {
 private:
  Float alpha;
  Float beta1;
  Float beta2;
  Float lambda;
  Float epsilon;

 public:
  AmsGradOptimizer() = delete;
  AmsGradOptimizer(const AmsGradOptimizer &) = delete;
  AmsGradOptimizer(const Params &optimizer_params, const Params &decay_params);
  virtual ~AmsGradOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

class AdamWOptimizer : public Optimizer {
 private:
  Float alpha;
  Float beta1;
  Float beta2;
  Float lambda;
  Float epsilon;

 public:
  AdamWOptimizer() = delete;
  AdamWOptimizer(const AdamWOptimizer &) = delete;
  AdamWOptimizer(const Params &optimizer_params, const Params &decay_params);
  virtual ~AdamWOptimizer();
  virtual int get_space(int dim);
  virtual void call(Float *data, Float *gds, int dim, u_int64_t global_step);
};

std::shared_ptr<Optimizer> get_optimizers(const Params &optimizer_params,
                                          const Params &decay_params);

#endif  // DAMO_EMBEDDING_OPTIMIZER_H
