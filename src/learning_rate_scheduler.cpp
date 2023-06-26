#include "learning_rate_scheduler.h"

Float exponential_decay(Float learning_rate, int64_t global_step,
                        const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  auto decay_rate = params.get<double>("decay_rate");
  return learning_rate * powf(decay_rate, global_step / decay_steps);
}

Float polynomial_decay(Float learning_rate, int64_t global_step,
                       const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  auto gstep = global_step < decay_steps ? global_step : decay_steps;
  auto end_learning_rate = params.get<double>("end_learning_rate", 0.0001);
  auto power = params.get<double>("power", 1.0);

  return (learning_rate - end_learning_rate) *
             powf(power, 1.0 - gstep / decay_steps) +
         end_learning_rate;
}

Float nature_exponential_decay(Float learning_rate, int64_t global_step,
                               const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  auto decay_rate = params.get<double>("decay_rate");
  return learning_rate * expf(-decay_rate * global_step / decay_steps);
}

Float inverse_time_decay(Float learning_rate, int64_t global_step,
                         const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  auto decay_rate = params.get<double>("decay_rate");
  return learning_rate / (1.0 + decay_rate * global_step / decay_steps);
}

Float cosine_decay(Float learning_rate, int64_t global_step,
                   const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  return learning_rate * 0.5 * (1.0 + cosf(M_PI * global_step / decay_steps));
}

Float liner_cosine_decay(Float learning_rate, int64_t global_step,
                         const Params &params) {
  auto alpha = params.get<double>("alpha", 0.0);
  auto beta = params.get<double>("beta", 0.001);
  auto decay_steps = params.get<double>("decay_steps");
  auto number_periods = params.get<double>("num_periods", 0.5);
  auto gstep = global_step < decay_steps ? global_step : decay_steps;
  auto liner_decay = (decay_steps - gstep) / decay_steps;
  auto coine_decay =
      -0.5 * (1.0 + cosf(M_PI * 2 * number_periods * gstep / decay_steps));
  auto decayed = (alpha + liner_decay) * coine_decay + beta;
  return learning_rate * decayed;
}

lr_scheduler get_lr_scheduler(const Params &p) {
  if (p.isnil()) {
    return nullptr;
  }
  auto name = p.get<std::string>("name");
  if (name == "exponential_decay") {
    return exponential_decay;
  } else if (name == "polynomial_decay") {
    return polynomial_decay;
  } else if (name == "nature_exponential_decay") {
    return nature_exponential_decay;
  } else if (name == "inverse_time_lr_scheduler") {
    return inverse_time_decay;
  } else if (name == "cosine_decay") {
    return cosine_decay;
  } else if (name == "liner_cosine_decay") {
    return liner_cosine_decay;
  } else {
    return nullptr;
  }
}
