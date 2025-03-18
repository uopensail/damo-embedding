#include "embedding.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace embedding {

// Embedding implementation ===========================================

Embedding::Embedding(const nlohmann::json& config) {
  constexpr const char* kRequiredFields[] = {"initializer", "optimizer", "dim",
                                             "group"};
  for (const auto& field : kRequiredFields) {
    if (!config.contains(field)) {
      throw std::invalid_argument("Missing required field in config: " +
                                  std::string(field));
    }
  }

  dim_ = config["dim"].get<int>();
  group_ = config["group"].get<int>();

  if (group_ < 0 || group_ >= kMaxEmbeddingNum) {
    throw std::out_of_range("Group index out of bounds: " +
                            std::to_string(group_));
  }

  initializer_ = get_initializers(Params{config["initializer"]});
  optimizer_ = get_optimizers(Params{config["optimizer"]});
}

// ApplyGradientsOperator implementation ==============================

ApplyGradientsOperator::ApplyGradientsOperator(const nlohmann::json& config) {
  if (!config.contains("embeddings") || !config["embeddings"].is_array()) {
    throw std::invalid_argument("Invalid embeddings configuration");
  }

  // embeddings_.fill(nullptr);  // Initialize all to nullptr

  for (const auto& entry : config["embeddings"]) {
    auto embedding = std::make_unique<Embedding>(entry);
    const int group = embedding->group();
    if (group < 0 || group >= kMaxEmbeddingNum) {
      throw std::out_of_range("Invalid embedding group: " +
                              std::to_string(group));
    }
    embeddings_[group] = std::move(embedding);
  }
}

bool ApplyGradientsOperator::FullMerge(
    const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
    const std::deque<std::string>& operand_list, std::string* new_value,
    rocksdb::Logger* logger) const {
  if (!existing_value || !new_value) {
    std::cerr << "Invalid merge arguments" << std::endl;
    return false;
  }

  const auto* metadata =
      reinterpret_cast<const MetaData*>(existing_value->data());
  if (metadata->group < 0 || metadata->group >= kMaxEmbeddingNum) {
    std::cerr << "Invalid group index: " << metadata->group << std::endl;
    return false;
  }

  const auto& embedding = embeddings_[metadata->group];
  if (!embedding) {
    std::cerr << "No embedding found for group: " << metadata->group
              << std::endl;
    return false;
  }

  // Create copy of existing value
  *new_value = existing_value->ToString();
  auto* new_metadata = reinterpret_cast<MetaData*>(new_value->data());

  try {
    for (const auto& operand : operand_list) {
      if (operand.size() != embedding->dimension() * sizeof(float)) {
        std::cerr << "Invalid gradient size for group " << metadata->group
                  << std::endl;
        return false;
      }

      const auto* grads = reinterpret_cast<const float*>(operand.data());
      embedding->optimizer()->call(new_metadata->data, grads,
                                   embedding->dimension(),
                                   ++new_metadata->update_num);
    }
  } catch (const std::exception& e) {
    std::cerr << "Optimization failed: " << e.what() << std::endl;
    return false;
  }

  new_metadata->update_time = get_current_time();
  return true;
}
// EmbeddingWarehouse implementation ==================================

EmbeddingWarehouse::EmbeddingWarehouse(const nlohmann::json& configure)
    : configure_(configure) {
  if (!configure_.contains("embeddings") ||
      !configure_["embeddings"].is_array()) {
    throw std::invalid_argument("Invalid embeddings configuration");
  }

  size_ = configure_["embeddings"].size();
  if (size_ <= 0 || size_ > kMaxEmbeddingNum) {
    throw std::out_of_range("Invalid number of embeddings: " +
                            std::to_string(size_));
  }

  for (const auto& entry : configure_["embeddings"]) {
    auto embedding = std::make_unique<Embedding>(entry);
    const int group = embedding->group();
    if (group < 0 || group >= kMaxEmbeddingNum) {
      throw std::out_of_range("Invalid embedding group: " +
                              std::to_string(group));
    }
    embeddings_[group] = std::move(embedding);
  }

  // Initialize RocksDB
  if (!configure_.contains("ttl") || !configure_.contains("dir")) {
    throw std::invalid_argument("Missing TTL or directory in config");
  }

  const int ttl = configure_["ttl"].get<int>();
  const std::string db_path = configure_["dir"].get<std::string>();

  rocksdb::Options options;
  options.create_if_missing = true;
  options.merge_operator.reset(new ApplyGradientsOperator(configure_));

  rocksdb::DBWithTTL* db_raw = nullptr;
  const rocksdb::Status status =
      rocksdb::DBWithTTL::Open(options, db_path, &db_raw, ttl);
  if (!status.ok()) {
    throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
  }

  db_ = RocksdbPtr(db_raw, RocksdbDeleter{});

  if (configure_.contains("reload_dir")) {
    load(configure_["reload_dir"].get<std::string>());
  }
}

void EmbeddingWarehouse::RocksdbDeleter::operator()(
    rocksdb::DBWithTTL* db) const {
  if (db) {
    // Perform cleanup operations
    const rocksdb::Status flush_status = db->Flush(rocksdb::FlushOptions());
    if (!flush_status.ok()) {
      std::cerr << "Failed to flush database: " << flush_status.ToString()
                << std::endl;
    }

    const rocksdb::Status compact_status =
        db->CompactRange(rocksdb::CompactRangeOptions(),
                         /*begin=*/nullptr,
                         /*end=*/nullptr);
    if (!compact_status.ok()) {
      std::cerr << "Failed to compact database: " << compact_status.ToString()
                << std::endl;
    }

    const rocksdb::Status close_status = db->Close();
    if (!close_status.ok()) {
      std::cerr << "Failed to close database: " << close_status.ToString()
                << std::endl;
    }

    delete db;
    std::cout << "RocksDB instance closed successfully" << std::endl;
  }
}

int EmbeddingWarehouse::dimension(int group) const {
  if (group < 0 || group >= kMaxEmbeddingNum) {
    throw std::out_of_range("Invalid group index: " + std::to_string(group));
  }
  if (!embeddings_[group]) {
    throw std::invalid_argument("No embedding for group: " +
                                std::to_string(group));
  }
  return embeddings_[group]->dimension();
}

void EmbeddingWarehouse::dump(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto snapshot_deleter = [this](const rocksdb::Snapshot* s) {
    db_->ReleaseSnapshot(s);
    ;
  };

  const auto snapshot =
      std::unique_ptr<const rocksdb::Snapshot, decltype(snapshot_deleter)>(
          db_->GetSnapshot(), snapshot_deleter);

  rocksdb::ReadOptions read_options;
  read_options.snapshot = snapshot.get();

  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(read_options));

  std::vector<int32_t> groups;
  std::vector<int32_t> group_dims;
  std::unordered_map<int32_t, size_t> group_indices;

  for (size_t i = 0; i < embeddings_.size(); ++i) {
    if (embeddings_[i]) {
      groups.push_back(embeddings_[i]->group());
      group_dims.push_back(embeddings_[i]->dimension());
      group_indices[embeddings_[i]->group()] = groups.size() - 1;
    }
  }

  std::ofstream writer(path, std::ios::binary);
  if (!writer) {
    throw std::runtime_error("Failed to open dump file: " + path);
  }

  // Write header
  const int num_groups = groups.size();
  writer.write(reinterpret_cast<const char*>(&num_groups), sizeof(int32_t));
  writer.write(reinterpret_cast<const char*>(groups.data()),
               sizeof(int32_t) * num_groups);
  writer.write(reinterpret_cast<const char*>(group_dims.data()),
               sizeof(int32_t) * num_groups);

  // Write data
  std::vector<int64_t> group_counts(num_groups, 0);
  writer.write(reinterpret_cast<const char*>(group_counts.data()),
               sizeof(int64_t) * num_groups);

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    const auto* metadata =
        reinterpret_cast<const MetaData*>(it->value().data());
    if (metadata->key == 0) continue;

    const auto index = group_indices.at(metadata->group);
    ++group_counts[index];

    writer.write(reinterpret_cast<const char*>(&metadata->key),
                 sizeof(int64_t));
    writer.write(reinterpret_cast<const char*>(&metadata->group),
                 sizeof(int32_t));
    writer.write(reinterpret_cast<const char*>(metadata->data),
                 sizeof(float) * metadata->dim);
  }

  // Update counts
  writer.seekp(sizeof(int32_t) + 2 * sizeof(int32_t) * num_groups);
  writer.write(reinterpret_cast<const char*>(group_counts.data()),
               sizeof(int64_t) * num_groups);
  writer.close();
}

void EmbeddingWarehouse::load(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Clear existing data
  const rocksdb::Slice begin, end;
  auto status = db_->DeleteRange(rocksdb::WriteOptions(),
                                 db_->DefaultColumnFamily(), begin, end);
  if (!status.ok()) {
    throw std::runtime_error("Failed to clear database: " + status.ToString());
  }

  // Load new data
  std::ifstream reader(path, std::ios::in | std::ios::binary);
  if (!reader) {
    throw std::runtime_error("Failed to open checkpoint file: " + path);
  }

  int64_t count = 0;
  reader.read(reinterpret_cast<char*>(&count), sizeof(count));

  constexpr size_t kInitialBufferSize = 4096;
  std::vector<char> key_buffer(kInitialBufferSize),
      value_buffer(kInitialBufferSize);

  for (int64_t i = 0; i < count; ++i) {
    size_t key_len = 0, value_len = 0;

    // Read key
    reader.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
    if (key_len > key_buffer.size()) {
      key_buffer.resize(key_len * 2);
    }
    reader.read(key_buffer.data(), key_len);

    // Read value
    reader.read(reinterpret_cast<char*>(&value_len), sizeof(value_len));
    if (value_len > value_buffer.size()) {
      value_buffer.resize(value_len * 2);
    }
    reader.read(value_buffer.data(), value_len);

    db_->Put(rocksdb::WriteOptions(),
             rocksdb::Slice(key_buffer.data(), key_len),
             rocksdb::Slice(value_buffer.data(), value_len));
  }
  reader.close();
  db_->Flush(rocksdb::FlushOptions());
  db_->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
}

std::unique_ptr<std::string> EmbeddingWarehouse::create_record(
    int group, int64_t key) const {
  // Validate group index
  if (group < 0 || group >= kMaxEmbeddingNum) {
    throw std::out_of_range("Invalid group index: " + std::to_string(group));
  }

  const auto& embedding = embeddings_[group];
  if (!embedding) {
    throw std::invalid_argument("No embedding found for group: " +
                                std::to_string(group));
  }

  const int dim = embedding->dimension();
  const int space_needed =
      sizeof(MetaData) + sizeof(float) * embedding->optimizer()->get_space(dim);

  auto value = std::make_unique<std::string>(space_needed, '\0');
  auto* metadata = reinterpret_cast<MetaData*>(value->data());

  // Initialize embedding data
  embedding->initializer()->call(metadata->data, dim);

  // Populate metadata fields
  metadata->update_num = 0;
  metadata->key = key;
  metadata->group = group;
  metadata->dim = dim;
  metadata->update_time = get_current_time();

  return value;
}

void EmbeddingWarehouse::checkpoint(const std::string& checkpoint_path) {
  // Acquire database snapshot with RAII management
  const auto snapshot_deleter = [this](const rocksdb::Snapshot* s) {
    db_->ReleaseSnapshot(s);
  };

  const auto snapshot =
      std::unique_ptr<const rocksdb::Snapshot, decltype(snapshot_deleter)>(
          db_->GetSnapshot(), snapshot_deleter);

  rocksdb::ReadOptions read_options;
  read_options.snapshot = snapshot.get();

  // Create iterator with RAII management
  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(read_options));

  try {
    std::ofstream writer(checkpoint_path, std::ios::binary);
    if (!writer) {
      throw std::runtime_error("Failed to open checkpoint file: " +
                               checkpoint_path);
    }

    int64_t record_count = 0;
    constexpr int64_t initial_count = 0;
    writer.write(reinterpret_cast<const char*>(&initial_count),
                 sizeof(initial_count));

    // Iterate through all database entries
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      const auto& key = it->key();
      const auto& value = it->value();

      // Write key metadata
      const size_t key_size = key.size();
      writer.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
      writer.write(key.data(), static_cast<std::streamsize>(key_size));

      // Write value metadata
      const size_t value_size = value.size();
      writer.write(reinterpret_cast<const char*>(&value_size),
                   sizeof(value_size));
      writer.write(value.data(), static_cast<std::streamsize>(value_size));

      ++record_count;
    }

    // Validate iteration results
    if (!it->status().ok()) {
      throw std::runtime_error("Database iteration failed: " +
                               it->status().ToString());
    }

    // Update record count at beginning of file
    writer.seekp(0);
    writer.write(reinterpret_cast<const char*>(&record_count), sizeof(int64_t));
    writer.close();
    std::cout << "Successfully created checkpoint with " << record_count
              << " entries" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Checkpoint creation failed: " << e.what() << std::endl;
    std::remove(checkpoint_path.c_str());  // Clean up partial file
    throw;
  }
}

void EmbeddingWarehouse::lookup(int group, const int64_t* keys, int len,
                                float* data, int n) const {
  // Validate input parameters
  if (group < 0 || group >= kMaxEmbeddingNum) {
    throw std::out_of_range("Invalid group index: " + std::to_string(group));
  }

  const auto& embedding = embeddings_[group];
  if (!embedding) {
    throw std::invalid_argument("No embedding found for group: " +
                                std::to_string(group));
  }

  const int dim = embedding->dimension();
  if (len * dim != n) {
    throw std::invalid_argument(
        "Dimension mismatch between input and data buffer");
  }

  std::vector<Key> group_keys(len);
  std::vector<rocksdb::Slice> slices;
  std::vector<std::string> results;

  // Prepare keys for batch operation
  for (int i = 0; i < len; ++i) {
    if (keys[i] == 0) continue;

    group_keys[i] = {group, keys[i]};
    slices.emplace_back(reinterpret_cast<const char*>(&group_keys[i]),
                        sizeof(Key));
  }

  // Execute batch read
  const rocksdb::ReadOptions read_options;
  const auto statuses = db_->MultiGet(read_options, slices, &results);

  rocksdb::WriteBatch batch;
  int result_index = 0;

  // Process query results
  for (int i = 0; i < len; ++i) {
    if (keys[i] == 0) continue;

    if (statuses[result_index].ok()) {
      const auto* metadata =
          reinterpret_cast<const MetaData*>(results[result_index].data());
      std::memcpy(&data[i * dim], metadata->data, sizeof(float) * dim);
    } else {
      auto value = create_record(group, keys[i]);
      const auto* metadata = reinterpret_cast<const MetaData*>(value->data());
      std::memcpy(&data[i * dim], metadata->data, sizeof(float) * dim);
      batch.Put(rocksdb::Slice(reinterpret_cast<const char*>(&group_keys[i]),
                               sizeof(Key)),
                *value);
    }
    ++result_index;
  }

  // Commit batch writes
  const rocksdb::WriteOptions write_options;
  const auto status = db_->Write(write_options, &batch);
  if (!status.ok()) {
    throw std::runtime_error("Failed to write batch: " + status.ToString());
  }
}

void EmbeddingWarehouse::apply_gradients(int group, const int64_t* keys,
                                         int len, const float* grads, int n) {
  // Validate input parameters
  if (group < 0 || group >= kMaxEmbeddingNum) {
    throw std::out_of_range("Invalid group index: " + std::to_string(group));
  }

  const auto& embedding = embeddings_[group];
  if (!embedding) {
    throw std::invalid_argument("No embedding found for group: " +
                                std::to_string(group));
  }

  const int dim = embedding->dimension();
  if (len * dim != n) {
    throw std::invalid_argument(
        "Dimension mismatch between input and gradients");
  }

  std::vector<Key> group_keys(len);
  rocksdb::WriteBatch batch;

  // Prepare merge operations
  for (int i = 0; i < len; ++i) {
    if (keys[i] == 0) continue;

    group_keys[i] = {group, keys[i]};
    const auto grad_slice = reinterpret_cast<const char*>(&grads[i * dim]);

    batch.Merge(rocksdb::Slice(reinterpret_cast<const char*>(&group_keys[i]),
                               sizeof(Key)),
                rocksdb::Slice(grad_slice, sizeof(float) * dim));
  }

  // Execute batch merge
  const rocksdb::WriteOptions write_options;
  const auto status = db_->Write(write_options, &batch);
  if (!status.ok()) {
    throw std::runtime_error("Failed to apply gradients: " + status.ToString());
  }
}
}  // namespace embedding
