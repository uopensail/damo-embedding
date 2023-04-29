#include "optimizer.h"

Optimizer::Optimizer(const Params &optimizer_params, const Params &scheduler)
    : name_(optimizer_params.get<std::string>("name")),
      function_(get_lr_scheduler(scheduler)), scheduler_(scheduler) {}

const std::string &Optimizer::get_name() { return name_; }

Optimizer::~Optimizer(){};

int Optimizer::get_space(int dim) { return dim; }

inline Float Optimizer::get_lr(u_int64_t global_step, Float learning_rate_) {
  if (function_) {
    return function_(learning_rate_, global_step, this->scheduler_);
  }
  return learning_rate_;
}

SGDOptimizer::SGDOptimizer(const Params &optimizer_params,
                           const Params &scheduler)
    : Optimizer(optimizer_params, scheduler),
      gamma_(optimizer_params.get<double>("gamma", 0.001)),
      lambda_(optimizer_params.get<double>("lambda", 0.0)) {}

SGDOptimizer::~SGDOptimizer() {}

void SGDOptimizer::call(Float *data, Float *gds, int dim,
                        u_int64_t global_step) {
  auto lr = get_lr(global_step, this->gamma_);
  for (int i = 0; i < dim; i++) {
    data[i] -=
        lr * (this->lambda_ != 0 ? gds[i] + this->lambda_ * data[i] : gds[i]);
  }
}

FTRLOptimizer::FTRLOptimizer(const Params &optimizer_params,
                             const Params &scheduler)
    : Optimizer(optimizer_params, scheduler),
      alpha_(optimizer_params.get<double>("alpha", 0.005)),
      beta_(optimizer_params.get<double>("beta", 0.0)),
      lambda1_(optimizer_params.get<double>("lambda1", 0.0)),
      lambda2_(optimizer_params.get<double>("lambda2", 0.0)) {}

FTRLOptimizer::~FTRLOptimizer() {}

int FTRLOptimizer::get_space(int dim) { return 3 * dim; }

void FTRLOptimizer::call(Float *data, Float *gds, int dim,
                         u_int64_t global_step) {
  Float *w = data;
  Float *z = &(data[dim]);
  Float *n = &(data[dim << 1]);
  Float tmp1, tmp2, delta, eta;
  for (int i = 0; i < dim; i++) {
    tmp1 = n[i] + gds[i] * gds[i];
    delta = ((safe_sqrt(tmp1) - safe_sqrt(n[i]))) / this->alpha_;
    z[i] += gds[i] - delta * w[i];
    tmp2 = abs(z[i]);
    if (tmp2 <= this->lambda1_) {
      w[i] = 0;
    } else {
      n[i] = tmp1;
      eta = -1.0 /
            ((this->beta_ + safe_sqrt(n[i])) / this->alpha_ + this->lambda2_);
      w[i] = eta * (z[i] - sign(z[i]) * this->lambda1_);
    }
  }
}

/**
 * @brief Construct a new Adam Optimizer:: Adam Optimizer object
 * Float gamma_;
  Float lambda_;
  Float eta_;
  Float epsilon_;
 *
 * @param optimizer_params
 * @param decay_params
 */

AdagradOptimizer::AdagradOptimizer(const Params &optimizer_params,
                                   const Params &scheduler)
    : Optimizer(optimizer_params, scheduler),
      gamma_(optimizer_params.get<double>("gamma", 0.01)),
      lambda_(optimizer_params.get<double>("lambda", 0.0)),
      eta_(optimizer_params.get<double>("eta", 0.0)),
      epsilon_(optimizer_params.get<double>("epsilon", 1e-10)) {}

AdagradOptimizer::~AdagradOptimizer() {}

/**
 * @brief get the space
 *
 * @param dim
 * @return int
 */
int AdagradOptimizer::get_space(int dim) { return 2 * dim; }

void AdagradOptimizer::call(Float *data, Float *gds, int dim,
                            u_int64_t global_step) {
  Float *w = data;
  Float *m = &(data[dim]);
  Float lr = get_lr(global_step, this->gamma_);
  Float tmp;
  for (int i = 0; i < dim; i++) {
    tmp = this->lambda_ != 0 ? gds[i] + this->lambda_ * w[i] : gds[i];
    m[i] += tmp * tmp;
    w[i] -= lr * tmp / (safe_sqrt(m[i]) + this->epsilon_);
  }
}

AdamOptimizer::AdamOptimizer(const Params &optimizer_params,
                             const Params &scheduler)
    : Optimizer(optimizer_params, scheduler),
      gamma_(optimizer_params.get<double>("gamma", 0.001)),
      beta1_(optimizer_params.get<double>("beta1", 0.9)),
      beta2_(optimizer_params.get<double>("beta2", 0.999)),
      lambda_(optimizer_params.get<double>("lambda", 0.0)),
      epsilon_(optimizer_params.get<double>("epsilon", 1e-8)) {}

AdamOptimizer::~AdamOptimizer() {}

/**
 * @brief get the space
 *
 * @param dim
 * @return int
 */
int AdamOptimizer::get_space(int dim) { return 3 * dim; }

void AdamOptimizer::call(Float *data, Float *gds, int dim,
                         u_int64_t global_step) {
  Float *w = data;
  Float *m = &(data[dim]);
  Float *v = &(data[dim << 1]);
  Float beta1_t = powf(this->beta1_, global_step);
  Float beta2_t = powf(this->beta2_, global_step);
  Float lr = get_lr(global_step, this->gamma_);
  Float tmp_gd, tmp_m, tmp_v;
  for (int i = 0; i < dim; i++) {
    tmp_gd = gds[i] + this->lambda_ * w[i];
    m[i] = this->beta1_ * m[i] + (1.0 - this->beta1_) * tmp_gd;
    v[i] = this->beta2_ * v[i] + (1.0 - this->beta2_) * tmp_gd * tmp_gd;
    tmp_m = m[i] / (1.0 - beta1_t);
    tmp_v = v[i] / (1.0 - beta2_t);
    w[i] -= lr * tmp_m / (safe_sqrt(tmp_v) + this->epsilon_);
  }
}

AmsGradOptimizer::AmsGradOptimizer(const Params &optimizer_params,
                                   const Params &scheduler)
    : Optimizer(optimizer_params, scheduler),
      gamma_(optimizer_params.get<double>("gamma", 0.001)),
      beta1_(optimizer_params.get<double>("beta1", 0.9)),
      beta2_(optimizer_params.get<double>("beta2", 0.999)),
      lambda_(optimizer_params.get<double>("lambda", 0.0)),
      epsilon_(optimizer_params.get<double>("epsilon", 1e-8)) {}

AmsGradOptimizer::~AmsGradOptimizer() {}

int AmsGradOptimizer::get_space(int dim) { return 4 * dim; }

void AmsGradOptimizer::call(Float *data, Float *gds, int dim,
                            u_int64_t global_step) {
  Float *w = data;
  Float *m = &(data[dim]);
  Float *v = &(data[dim * 2]);
  Float *v_max = &(data[dim * 3]);
  Float beta1_t = powf(this->beta1_, global_step);
  Float beta2_t = powf(this->beta2_, global_step);
  Float lr = get_lr(global_step, this->gamma_);
  Float tmp_gd, tmp_m, tmp_v;
  for (int i = 0; i < dim; i++) {
    tmp_gd = this->lambda_ != 0 ? gds[i] + this->lambda_ * w[i] : gds[i];
    m[i] = this->beta1_ * m[i] + (1.0 - this->beta1_) * tmp_gd;
    v[i] = this->beta2_ * v[i] + (1.0 - this->beta2_) * tmp_gd * tmp_gd;
    tmp_m = m[i] / (1.0 - beta1_t);
    tmp_v = v[i] / (1.0 - beta2_t);
    v_max[i] = v_max[i] < tmp_v ? tmp_v : v_max[i];
    w[i] -= lr * tmp_m / (safe_sqrt(v_max[i]) + this->epsilon_);
  }
}

AdamWOptimizer::AdamWOptimizer(const Params &optimizer_params,
                               const Params &scheduler)
    : Optimizer(optimizer_params, scheduler),
      gamma_(optimizer_params.get<double>("gamma", 0.001)),
      beta1_(optimizer_params.get<double>("beta1", 0.9)),
      beta2_(optimizer_params.get<double>("beta2", 0.999)),
      lambda_(optimizer_params.get<double>("lambda", 0.01)),
      epsilon_(optimizer_params.get<double>("epsilon", 1e-8)) {}

AdamWOptimizer::~AdamWOptimizer() {}

int AdamWOptimizer::get_space(int dim) { return 3 * dim; }

void AdamWOptimizer::call(Float *data, Float *gds, int dim,
                          u_int64_t global_step) {
  Float *w = data;
  Float *m = &(data[dim]);
  Float *v = &(data[dim << 1]);
  Float beta1_t = powf(this->beta1_, global_step);
  Float beta2_t = powf(this->beta2_, global_step);
  Float lr = get_lr(global_step, this->gamma_);
  Float tmp_m, tmp_v;

  for (int i = 0; i < dim; i++) {
    w[i] -= this->lambda_ * lr * w[i];
    m[i] = this->beta1_ * m[i] + (1.0 - this->beta1_) * gds[i];
    v[i] = this->beta2_ * v[i] + (1.0 - this->beta2_) * gds[i] * gds[i];
    tmp_m = m[i] / (1.0 - beta1_t);
    tmp_v = v[i] / (1.0 - beta2_t);
    w[i] -= lr * tmp_m / (safe_sqrt(tmp_v) + this->epsilon_);
  }
}

LionOptimizer::LionOptimizer(const Params &optimizer_params,
                             const Params &scheduler)
    : Optimizer(optimizer_params, scheduler),
      eta_(optimizer_params.get<double>("eta", 0.0003)),
      beta1_(optimizer_params.get<double>("beta1", 0.9)),
      beta2_(optimizer_params.get<double>("beta2", 0.99)),
      lambda_(optimizer_params.get<double>("lambda", 0.01)) {}

LionOptimizer::~LionOptimizer() {}

int LionOptimizer::get_space(int dim) { return 2 * dim; }

void LionOptimizer::call(Float *data, Float *gds, int dim,
                         u_int64_t global_step) {
  Float *w = data;
  Float *m = &(data[dim]);
  Float lr = get_lr(global_step, this->eta_);
  Float tmp_mu;
  for (int i = 0; i < dim; i++) {
    tmp_mu = sign(this->beta1_ * m[i] + (1.0 - this->beta1_) * gds[i]) +
             w[i] * this->lambda_;
    w[i] -= lr * tmp_mu;
    m[i] = this->beta2_ * m[i] + (1.0 - this->beta2_) * gds[i];
  }
}

std::shared_ptr<Optimizer> get_optimizers(const Params &optimizer_params,
                                          const Params &scheduler) {
  if (optimizer_params.isnil()) {
    std::cerr << "optimizer params is nil" << std::endl;
    exit(0);
  }
  auto name = optimizer_params.get<std::string>("name", "sgd");
  if (name == "sgd") {
    return std::shared_ptr<Optimizer>(
        new SGDOptimizer{optimizer_params, scheduler});
  } else if (name == "ftrl") {
    return std::shared_ptr<Optimizer>(
        new FTRLOptimizer{optimizer_params, scheduler});
  } else if (name == "adam") {
    return std::shared_ptr<Optimizer>(
        new AdamOptimizer{optimizer_params, scheduler});
  } else if (name == "amsgrad") {
    return std::shared_ptr<Optimizer>(
        new AmsGradOptimizer{optimizer_params, scheduler});
  } else if (name == "adamw") {
    return std::shared_ptr<Optimizer>(
        new AdamWOptimizer{optimizer_params, scheduler});
  } else if (name == "lion") {
    return std::shared_ptr<Optimizer>(
        new LionOptimizer{optimizer_params, scheduler});
  } else {
    std::cout << "No Such Optimizer: " << name << std::endl;
    exit(-3);
  }
}
