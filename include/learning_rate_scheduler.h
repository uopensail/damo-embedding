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

#ifndef DAMO_EMBEDDING_LEARNING_RATE_SCHEDULER_H
#define DAMO_EMBEDDING_LEARNING_RATE_SCHEDULER_H

#include <functional>

#include "common.h"

using lr_scheduler = std::function<Float(
    Float learning_rate, int64_t global_step, const Params &params)>;

Float exponential_decay(Float learning_rate, int64_t global_step,
                        const Params &params);
Float polynomial_decay(Float learning_rate, int64_t global_step,
                       const Params &params);
Float nature_exponential_decay(Float learning_rate, int64_t global_step,
                               const Params &params);
Float inverse_time_decay(Float learning_rate, int64_t global_step,
                         const Params &params);
Float cosine__decay(Float learning_rate, int64_t global_step,
                    const Params &params);
Float liner_cosine_decay(Float learning_rate, int64_t global_step,
                         const Params &params);
lr_scheduler get_lr_scheduler(const Params &p);

#endif  // DAMO_EMBEDDING_LEARNING_RATE_SCHEDULER_H
