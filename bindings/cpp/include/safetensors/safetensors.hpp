/*
 * Copyright (c) 2025 Dapeng Feng
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "nlohmann/json.hpp"
#include "rust/cxx.h"
#include "safetensors/mmap.hpp"
#include "safetensors_abi/lib.h"

namespace safetensors {

constexpr std::size_t N_LEN = 8;

class SafeOpen {
 public:
  struct TensorView {
    std::vector<std::size_t> shape;
    Dtype dtype;
    const void* data_ptr;
    std::size_t data_len = 0;
  };

  explicit SafeOpen(const std::string& filename) {
    file_ptr_ = std::make_unique<File>(File::SafeOpen(filename));
    mmap_ptr_ = std::make_unique<Mmap>(Mmap::SafeMap(file_ptr_.get()));

    if (mmap_ptr_->size() < N_LEN) {
      throw std::runtime_error(
          fmt::format("{}:{} file {} is too small: {} < {}", __FILE__, __LINE__,
                      filename, mmap_ptr_->size(), N_LEN));
    }

    buffer_ = rust::Slice<std::uint8_t const>(*mmap_ptr_);

    tensor_views_ = deserialize(buffer_);
    metadata_ = metadata(buffer_);
  }

  SafeOpen(const SafeOpen&) = delete;
  SafeOpen& operator=(const SafeOpen&) = delete;

  SafeOpen(SafeOpen&&) = default;
  SafeOpen& operator=(SafeOpen&&) = default;

  ~SafeOpen() = default;

  std::vector<std::string> keys() const {
    std::vector<std::string> keys;
    for (const auto& pair : tensor_views_) {
      std::string key(pair.key.data(), pair.key.size());
      keys.emplace_back(std::move(key));
    }
    return keys;
  }

  TensorView get_tensor(const rust::String& key) const {
    for (const auto& pair : tensor_views_) {
      if (pair.key == key) {
        std::vector<std::size_t> shape(pair.value.shape.data(),
                                                pair.value.shape.data() +
                                                    pair.value.shape.size());
        Dtype dtype = pair.value.dtype;
        const void* data_ptr = pair.value.data.data();
        std::size_t data_len = pair.value.data_len;

        return TensorView{std::move(shape), dtype, data_ptr, data_len};
      }
    }
    throw std::runtime_error(fmt::format("{}:{} key '{}' not found", __FILE__,
                                         __LINE__, key));
  }

  nlohmann::ordered_map<std::string, std::string> get_metadata() const {
    nlohmann::ordered_map<std::string, std::string> metadata_map;
    for (const auto& pair : metadata_) {
      std::string key(pair.key.data(), pair.key.size());
      std::string value(pair.value.data(), pair.value.size());
      metadata_map[key] = value;
    }
    return metadata_map;
  }

 private:
  std::unique_ptr<File> file_ptr_;
  std::unique_ptr<Mmap> mmap_ptr_;
  rust::Slice<std::uint8_t const> buffer_;
  rust::Vec<PairStrTensorView> tensor_views_;
  rust::Vec<PairStrStr> metadata_;
};


}  // namespace safetensors
