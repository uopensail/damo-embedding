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

#ifndef DAMO_EMBEDDING_INITIALIZER_H
#define DAMO_EMBEDDING_INITIALIZER_H

#pragma once
#include <iostream>

#include "common.h"

class Initializer {
public:
  Initializer() = delete;
  Initializer(const Initializer &) = delete;
  Initializer(const Params &initializer_params);
  virtual ~Initializer();
  const std::string &get_name();

  /**
   * @brief initialize the weights
   *
   * @param w weights to be initialized
   * @param wn width of the weights
   */
  virtual void call(Float *data, int dim) = 0;

protected:
  std::string name_;
};

class Zeros : public Initializer {
public:
  Zeros() = delete;
  Zeros(const Zeros &) = delete;
  Zeros(const Params &initializer_params);
  virtual ~Zeros();
  virtual void call(Float *data, int dim);
};

class Ones : public Initializer {
public:
  Ones() = delete;
  Ones(const Ones &) = delete;
  Ones(const Params &initializer_params);
  virtual ~Ones();
  virtual void call(Float *data, int dim);
};

class RandomUniform : public Initializer {
public:
  RandomUniform() = delete;
  RandomUniform(const RandomUniform &) = delete;
  RandomUniform(const Params &initializer_params);
  virtual ~RandomUniform();
  virtual void call(Float *data, int dim);

private:
  double min_;
  double max_;
  std::uniform_real_distribution<double> distribution;
  std::default_random_engine random;
};

class RandomNormal : public Initializer {
public:
  RandomNormal() = delete;
  RandomNormal(const RandomNormal &) = delete;
  RandomNormal(const Params &initializer_params);
  virtual ~RandomNormal();
  virtual void call(Float *data, int dim);

private:
  double mean_;
  double stddev_;
  std::normal_distribution<double> distribution;
  std::default_random_engine random;
};

class TruncateNormal : public Initializer {
public:
  TruncateNormal() = delete;
  TruncateNormal(const TruncateNormal &) = delete;
  TruncateNormal(const Params &initializer_params);
  virtual ~TruncateNormal();
  virtual void call(Float *data, int dim);

private:
  double mean_;
  double stddev_;
  std::normal_distribution<double> distribution;
  std::default_random_engine random;
};

std::shared_ptr<Initializer> get_initializers(const Params &p);
#endif // DAMO_EMBEDDING_INITIALIZER_H
