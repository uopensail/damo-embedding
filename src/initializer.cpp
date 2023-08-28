#include "initializer.h"

#include <iostream>
Initializer::Initializer(const Params &initializer_params)
    : name_(initializer_params.get<std::string>("name")) {}

Initializer::~Initializer() {}

const std::string &Initializer::get_name() { return name_; }

Zeros::Zeros(const Params &initializer_params)
    : Initializer(initializer_params) {}

std::string Zeros::to_string() const { return "zeros"; }

Zeros::~Zeros() {}

void Zeros::call(Float *data, int dim) { memset(data, 0, sizeof(Float) * dim); }

Ones::Ones(const Params &initializer_params)
    : Initializer(initializer_params) {}

std::string Ones::to_string() const { return "ones"; }

Ones::~Ones() {}

void Ones::call(Float *data, int dim) {
  for (int i = 0; i < dim; i++) {
    data[i] = 1.0f;
  }
}

RandomUniform::RandomUniform(const Params &initializer_params)
    : Initializer(initializer_params),
      min_(initializer_params.get<double>("min", -1.0)),
      max_(initializer_params.get<double>("max", 1.0)),
      distribution(min_, max_),
      random(time(NULL)) {
  assert(max_ > min_);
}

std::string RandomUniform::to_string() const {
  return "random_uniform: [" + std::to_string(min_) + ", " +
         std::to_string(max_) + "]";
}

RandomUniform::~RandomUniform() {}

void RandomUniform::call(Float *data, int dim) {
  for (int i = 0; i < dim; i++) {
    data[i] = distribution(random);
  }
}

RandomNormal::RandomNormal(const Params &initializer_params)
    : Initializer(initializer_params),
      mean_(initializer_params.get<double>("mean", 0.0)),
      stddev_(initializer_params.get<double>("stddev", 1.0)),
      distribution(mean_, stddev_),
      random(time(NULL)) {
  assert(stddev_ > 0.0);
}

std::string RandomNormal::to_string() const {
  return "random_normal: (mean=" + std::to_string(mean_) +
         ", std=" + std::to_string(stddev_) + ")";
}

RandomNormal::~RandomNormal() {}

void RandomNormal::call(Float *data, int dim) {
  for (int i = 0; i < dim; i++) {
    data[i] = distribution(random);
  }
}

TruncateNormal::TruncateNormal(const Params &initializer_params)
    : Initializer(initializer_params),
      mean_(initializer_params.get<double>("mean", 0.0)),
      stddev_(initializer_params.get<double>("stddev", 1.0)),
      distribution(mean_, stddev_),
      random(time(NULL)) {
  assert(stddev_ > 0.0);
}

std::string TruncateNormal::to_string() const {
  return "truncate_normal: (mean=" + std::to_string(mean_) +
         ", std=" + std::to_string(stddev_) + ")";
}

TruncateNormal::~TruncateNormal() {}

void TruncateNormal::call(Float *data, int dim) {
  double tmp;
  for (int i = 0; i < dim; i++) {
    tmp = distribution(random);
    while (abs(tmp - mean_) > 2.0 * stddev_) {
      tmp = distribution(random);
    }
    data[i] = tmp;
  }
}

std::shared_ptr<Initializer> get_initializers(const Params &p) {
  if (p.isnil()) {
    std::cerr << "initializer params is nil" << std::endl;
    exit(0);
  }
  auto name = p.get<std::string>("name", "zeros");
  if (name == "zeros") {
    return std::shared_ptr<Initializer>{new Zeros(p)};
  } else if (name == "ones") {
    return std::shared_ptr<Initializer>{new Ones(p)};
  } else if (name == "random_uniform") {
    return std::shared_ptr<Initializer>{new RandomUniform(p)};
  } else if (name == "random_normal") {
    return std::shared_ptr<Initializer>{new RandomNormal(p)};
  } else if (name == "truncate_normal") {
    return std::shared_ptr<Initializer>{new TruncateNormal(p)};
  } else {
    std::cout << "No Such Initializer: " << name << std::endl;
    exit(-2);
  }
}