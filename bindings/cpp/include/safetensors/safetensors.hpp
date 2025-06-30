/*
 * Copyright (c) 2025 Dapeng Feng
 * All rights reserved.
 */

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "fmt/format.h"
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
    const void* data_ptr = nullptr;
    std::size_t data_len = 0;
  };

  explicit SafeOpen(const std::string& filename)
      : file_ptr_(std::make_unique<File>(filename)) {
    mmap_ptr_ = std::make_unique<Mmap>(file_ptr_.get());

    if (mmap_ptr_->size() < N_LEN) {
      throw std::runtime_error(
          fmt::format("{}:{} file {} is too small: {} < {}", __FILE__, __LINE__,
                      filename, mmap_ptr_->size(), N_LEN));
    }

    buffer_ = rust::Slice<std::uint8_t const>(*mmap_ptr_);

    tensor_views_ = deserialize(buffer_);
    metadata_ = metadata(buffer_);

    metadata_map_.reserve(metadata_.size());
    for (const auto& pair : metadata_) {
      std::string key(pair.key.data(), pair.key.size());
      std::string value(pair.value.data(), pair.value.size());
      metadata_map_[std::move(key)] = std::move(value);
    }

    std::vector<std::pair<std::string, TensorView>> tensor_views_vector;

    tensor_views_vector.reserve(tensor_views_.size());
    for (const auto& pair : tensor_views_) {
      std::vector<std::size_t> shape(
          pair.value.shape.begin(),
          pair.value.shape.end());
      Dtype dtype = pair.value.dtype;
      const void* data_ptr = pair.value.data.data();
      std::size_t data_len = pair.value.data_len;
      std::string key(pair.key.data(), pair.key.size());
      // TODO(dp): check the access order 
      tensor_views_vector.emplace_back(
          std::make_pair(std::move(key), TensorView{std::move(shape), dtype, data_ptr,
                                         data_len}));
      // tensor_views_map_[key] =
      //     TensorView{std::move(shape), dtype, data_ptr, data_len,
      //                std::move(data_offsets)};
      // keys_.emplace_back(std::move(key));
    }

    std::sort(tensor_views_vector.begin(), tensor_views_vector.end(),
              [](const auto& a, const auto& b) {
                return a.second.data_ptr < b.second.data_ptr;
              });

    for (const auto& pair : tensor_views_vector) {
      // Store the tensor view in the map and keys vector
      tensor_views_map_[pair.first] = pair.second;
      keys_.emplace_back(pair.first);
    }
  }

  SafeOpen(const SafeOpen&) = delete;
  SafeOpen& operator=(const SafeOpen&) = delete;

  SafeOpen(SafeOpen&&) = default;
  SafeOpen& operator=(SafeOpen&&) = default;

  ~SafeOpen() = default;

  inline std::vector<std::string> keys() const noexcept{
    return keys_;
  }

  TensorView get_tensor(const std::string& key) {
    if (tensor_views_map_.find(key) == tensor_views_map_.end())
      throw std::runtime_error(fmt::format("{}:{} key '{}' not found", __FILE__,
                                           __LINE__, std::string(key)));
    return tensor_views_map_[key];
  }

  inline std::unordered_map<std::string, std::string> get_metadata() const noexcept {
    return metadata_map_;
  }

 private:
  std::unique_ptr<File> file_ptr_;
  std::unique_ptr<Mmap> mmap_ptr_;
  rust::Slice<std::uint8_t const> buffer_;
  rust::Vec<PairStrTensorView> tensor_views_;
  rust::Vec<PairStrStr> metadata_;
  std::vector<std::string> keys_;
  std::unordered_map<std::string, TensorView> tensor_views_map_;
  std::unordered_map<std::string, std::string> metadata_map_;
};

}  // namespace safetensors
