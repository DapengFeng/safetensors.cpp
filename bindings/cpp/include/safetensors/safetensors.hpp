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

#include "fmt/format.h"
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
    const void* data_ptr = nullptr;
    std::size_t data_len = 0;
    std::pair<std::size_t, std::size_t> data_offsets;
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

    for (const auto& pair : metadata_) {
      std::string key(pair.key.data(), pair.key.size());
      std::string value(pair.value.data(), pair.value.size());
      metadata_map_[key] = value;
    }

    for (const auto& pair : tensor_views_) {
      std::vector<std::size_t> shape(
          pair.value.shape.begin(),
          pair.value.shape.end());
      Dtype dtype = pair.value.dtype;
      const void* data_ptr = pair.value.data.data();
      std::size_t data_len = pair.value.data_len;
      std::string key(pair.key.data(), pair.key.size());
      std::pair<std::size_t, std::size_t> data_offsets(
          pair.value.data_offsets[0], pair.value.data_offsets[1]);
      // TODO(dp): check the access order
      tensor_views_map_[key] =
          TensorView{std::move(shape), dtype, data_ptr, data_len,
                     std::move(data_offsets)};
      // tensor_views_vector_.emplace_back(
      //     std::make_pair(key, TensorView{std::move(shape), dtype, data_ptr,
      //                                    data_len, std::move(data_offsets)}));
    }

    // std::sort(tensor_views_vector_.begin(), tensor_views_vector_.end(),
    //           [](const auto& a, const auto& b) {
    //             return a.second.data_ptr <
    //                    b.second.data_ptr;
    //           });

    // for (const auto& pair : tensor_views_vector_) {
    //   tensor_views_map_[pair.first] = pair.second;
    // }
  }

  SafeOpen(const SafeOpen&) = delete;
  SafeOpen& operator=(const SafeOpen&) = delete;

  SafeOpen(SafeOpen&&) = default;
  SafeOpen& operator=(SafeOpen&&) = default;

  ~SafeOpen() = default;

  std::vector<std::string> keys() const {
    std::vector<std::string> keys;
    for (const auto& pair : tensor_views_map_) {
      std::string key(pair.first.data(), pair.first.size());
      keys.emplace_back(std::move(key));
    }
    return keys;
  }

  TensorView get_tensor(const std::string& key) const {
    if (tensor_views_map_.find(key) == tensor_views_map_.end())
      throw std::runtime_error(fmt::format("{}:{} key '{}' not found", __FILE__,
                                           __LINE__, std::string(key)));
    return tensor_views_map_[key];
  }

  nlohmann::ordered_map<std::string, std::string> get_metadata() const {
    return metadata_map_;
  }

 private:
  std::unique_ptr<File> file_ptr_;
  std::unique_ptr<Mmap> mmap_ptr_;
  rust::Slice<std::uint8_t const> buffer_;
  rust::Vec<PairStrTensorView> tensor_views_;
  rust::Vec<PairStrStr> metadata_;
  // std::vector<std::pair<std::string, TensorView>> tensor_views_vector_;
  nlohmann::ordered_map<std::string, TensorView> tensor_views_map_;
  nlohmann::ordered_map<std::string, std::string> metadata_map_;
};

}  // namespace safetensors
