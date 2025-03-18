#include "damo.h"

// Constructor implementation example
PyDamo::PyDamo(const std::string &config_file) {
  try {
    std::ifstream config_stream(config_file);
    nlohmann::json config;
    config_stream >> config;
    warehouse_ = std::make_shared<embedding::EmbeddingWarehouse>(config);
  } catch (const std::exception &e) {
    throw py::value_error("Failed to initialize PyDamo: " +
                          std::string(e.what()));
  }
}

void PyDamo::pull(int group, py::array_t<int64_t> embedding_keys,
                  py::array_t<float> embedding_weights) {
  // Validate input buffers
  const py::buffer_info keys_info = embedding_keys.request();
  const py::buffer_info weights_info = embedding_weights.request();

  // Check array dimensions
  if (keys_info.ndim != 1 || weights_info.ndim != 1) {
    throw py::value_error("Both keys and weights arrays must be 1-dimensional");
  }

  // Check array writeability
  if (!embedding_weights.writeable()) {
    throw py::value_error("Weights array must be writable");
  }

  // Validate array sizes
  const size_t num_keys = keys_info.size;
  const size_t weights_size = weights_info.size;
  if (num_keys == 0) {
    throw py::value_error("Keys array cannot be empty");
  }

  try {
    // Get buffer pointers with const correctness
    const auto *keys_ptr = static_cast<const int64_t *>(keys_info.ptr);
    auto *weights_ptr = static_cast<float *>(weights_info.ptr);

    // Verify warehouse initialization
    if (!warehouse_) {
      throw std::runtime_error("Embedding warehouse not initialized");
    }

    // Perform lookup operation
    warehouse_->lookup(group, keys_ptr, static_cast<int>(num_keys), weights_ptr,
                       static_cast<int>(weights_size));
  } catch (const std::exception &e) {
    // Convert C++ exceptions to Python exceptions
    throw py::value_error("Embedding lookup failed: " + std::string(e.what()));
  }
}

void PyDamo::push(int group, py::array_t<int64_t> keys,
                  py::array_t<float> gradients) {
  // Validate input buffers
  const auto keys_info = keys.request();
  const auto grads_info = gradients.request();

  // Check array dimensions
  if (keys_info.ndim != 1) {
    throw std::invalid_argument("Keys array must be 1-dimensional");
  }
  if (grads_info.ndim != 1) {
    throw std::invalid_argument("Gradients array must be 1-dimensional");
  }

  // Check array sizes
  const size_t num_keys = keys_info.size;
  const size_t grads_size = grads_info.size;
  if (num_keys == 0) {
    throw std::invalid_argument("Empty keys array");
  }

  try {
    // Get buffer pointers with bounds checking
    const auto *keys_ptr = static_cast<const int64_t *>(keys_info.ptr);
    const auto *grads_ptr = static_cast<const float *>(grads_info.ptr);

    // Verify warehouse exists
    if (!warehouse_) {
      throw std::runtime_error("Embedding warehouse not initialized");
    }

    // Apply gradients through warehouse
    warehouse_->apply_gradients(group, keys_ptr, static_cast<int>(num_keys),
                                grads_ptr, static_cast<int>(grads_size));
  } catch (const std::exception &e) {
    // Convert C++ exceptions to Python exceptions
    throw py::value_error("Failed to apply gradients: " +
                          std::string(e.what()));
  }
}
