#include "learning_rate_scheduler.h"

Float exponential_lr_scheduler(Float learning_rate, u_int64_t global_step,
                               const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  auto decay_rate = params.get<double>("decay_rate");
  return learning_rate -
         learning_rate * powf(decay_rate, global_step / decay_steps);
}

Float polynomial_lr_scheduler(Float learning_rate, u_int64_t global_step,
                              const Params &params) {
  auto end_learning_rate = params.get<double>("end_learning_rate");
  auto power = params.get<double>("power");
  auto decay_steps = params.get<double>("decay_steps");
  auto gstep = global_step < decay_steps ? global_step : decay_steps;
  return learning_rate -
         (learning_rate - end_learning_rate) *
             powf(power, 1.0 - gstep / decay_steps) +
         end_learning_rate;
}

Float nature_exp_lr_scheduler(Float learning_rate, u_int64_t global_step,
                              const Params &params) {
  auto decay_rate = params.get<double>("decay_rate");
  return learning_rate - learning_rate * expf(-decay_rate * global_step);
}

Float inverse_time_lr_scheduler(Float learning_rate, u_int64_t global_step,
                                const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  auto decay_rate = params.get<double>("decay_rate");
  return learning_rate -
         learning_rate / (1.0 + decay_rate * global_step / decay_steps);
}

Float cosine_lr_scheduler(Float learning_rate, u_int64_t global_step,
                          const Params &params) {
  auto decay_steps = params.get<double>("decay_steps");
  auto gstep = global_step < decay_steps ? global_step : decay_steps;
  return learning_rate -
         learning_rate * 0.5 * (1.0 + cosf(M_PI * gstep / decay_steps));
}

Float liner_cosine_lr_scheduler(Float learning_rate, u_int64_t global_step,
                                const Params &params) {
  auto alpha = params.get<double>("alpha");
  auto beta = params.get<double>("beta");
  auto decay_steps = params.get<double>("decay_steps");
  auto number_periods = params.get<double>("number_periods");
  auto gstep = global_step < decay_steps ? global_step : decay_steps;
  auto liner_decay = (decay_steps - gstep) / decay_steps;
  auto coine_decay =
      -.5 * (1.0 + cosf(M_PI * 2 * number_periods * gstep / decay_steps));
  auto decayed = (alpha + liner_decay) * coine_decay + beta;
  return learning_rate - learning_rate * decayed;
}

lr_scheduler get_lr_scheduler(const Params &p) {
  auto name = p.get<std::string>("name");
  if (name == "exponential_lr_scheduler") {
    return exponential_lr_scheduler;
  } else if (name == "polynomial_lr_scheduler") {
    return polynomial_lr_scheduler;
  } else if (name == "nature_exp_lr_scheduler") {
    return nature_exp_lr_scheduler;
  } else if (name == "inverse_time_lr_scheduler") {
    return inverse_time_lr_scheduler;
  } else if (name == "cosine_lr_scheduler") {
    return cosine_lr_scheduler;
  } else if (name == "liner_cosine_lr_scheduler") {
    return liner_cosine_lr_scheduler;
  } else {
    return nullptr;
  }
}
