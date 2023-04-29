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

#ifndef DAMO_EMBEDDING_LEARNING_RATE_SCHEDULER_H
#define DAMO_EMBEDDING_LEARNING_RATE_SCHEDULER_H

#include <functional>

#include "common.h"

using lr_scheduler = std::function<Float(
    Float learning_rate, u_int64_t global_step, const Params &params)>;

Float exponential_decay(Float learning_rate, u_int64_t global_step,
                        const Params &params);
Float polynomial_decay(Float learning_rate, u_int64_t global_step,
                       const Params &params);
Float nature_exponential_decay(Float learning_rate, u_int64_t global_step,
                               const Params &params);
Float inverse_time_decay(Float learning_rate, u_int64_t global_step,
                         const Params &params);
Float cosine__decay(Float learning_rate, u_int64_t global_step,
                    const Params &params);
Float liner_cosine_decay(Float learning_rate, u_int64_t global_step,
                         const Params &params);
lr_scheduler get_lr_scheduler(const Params &p);

#endif // DAMO_EMBEDDING_LEARNING_RATE_SCHEDULER_H
