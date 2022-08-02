#ifndef DAMO_EMBEDDING_DECAY_LEARNING_RATE_H
#define DAMO_EMBEDDING_DECAY_LEARNING_RATE_H

#include "common.h"

using decay_lr_func = Float (*)(Float learning_rate, long long global_step, const Params &params);

Float exponential_decay(Float learning_rate, long long global_step, const Params &params);
Float polynomial_decay(Float learning_rate, long long global_step, const Params &params);
Float nature_exp_decay(Float learning_rate, long long global_step, const Params &params);
Float inverse_time_decay(Float learning_rate, long long global_step, const Params &params);
Float cosine_decay(Float learning_rate, long long global_step, const Params &params);
Float liner_cosine_decay(Float learning_rate, long long global_step, const Params &params);
decay_lr_func get_decay_lr_func(const Params &p);

#endif // DAMO_EMBEDDING_DECAY_LEARNING_RATE_H
