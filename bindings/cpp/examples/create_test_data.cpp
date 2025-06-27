/*
 * Copyright (c) 2025 Dapeng Feng
 * All rights reserved.
 */

// SafeTensors C++ Example: Create Test Data
// This example demonstrates how to create test safetensors data
// that can be used with the safe_open_example.

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "nlohmann/json.hpp"
#include "safetensors/safetensors.hpp"

// Simple function to create a test safetensors file
void create_test_safetensors(const std::string& filename) {
  // Create some test data
  std::vector<float> tensor1_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int32_t> tensor2_data = {10, 20, 30, 40};

  // Create metadata and tensor info
  nlohmann::ordered_json header;

  // Add metadata
  header["__metadata__"] = nlohmann::ordered_json::object();
  header["__metadata__"]["created_by"] = "SafeTensors C++ Example";
  header["__metadata__"]["version"] = "1.0";

  // Add tensor1 info (shape: [2, 3], dtype: F32)
  header["tensor1"] = nlohmann::ordered_json::object();
  header["tensor1"]["dtype"] = "F32";
  header["tensor1"]["shape"] = nlohmann::ordered_json::array({2, 3});
  header["tensor1"]["data_offsets"] =
      nlohmann::ordered_json::array({0, tensor1_data.size() * sizeof(float)});

  // Add tensor2 info (shape: [4], dtype: I32)
  std::size_t tensor2_offset = tensor1_data.size() * sizeof(float);
  header["tensor2"] = nlohmann::ordered_json::object();
  header["tensor2"]["dtype"] = "I32";
  header["tensor2"]["shape"] = nlohmann::ordered_json::array({4});
  header["tensor2"]["data_offsets"] = nlohmann::ordered_json::array(
      {tensor2_offset, tensor2_offset + tensor2_data.size() * sizeof(int32_t)});

  rust::Vec<safetensors::PairStrStr> metadata;
  // cppcheck-suppress unassignedVariable
  for (const auto& [key, value] : header["__metadata__"].items()) {
    metadata.emplace_back(
        safetensors::PairStrStr{key, value.get<std::string>()});
  }
  rust::Vec<safetensors::PairStrTensorView> tensors;
  // cppcheck-suppress unassignedVariable
  for (const auto& [key, value] : header.items()) {
    if (key == "__metadata__") continue;  // Skip metadata
    safetensors::TensorView tensor_view;
    if (key == "tensor1") {
      tensor_view.dtype = safetensors::Dtype::F32;
      tensor_view.shape = rust::Vec<std::size_t>({2, 3});
      tensor_view.data = rust::Slice<std::uint8_t const>(
          reinterpret_cast<const std::uint8_t*>(tensor1_data.data()),
          tensor1_data.size() * sizeof(float));
    } else if (key == "tensor2") {
      tensor_view.dtype = safetensors::Dtype::I32;
      tensor_view.shape = rust::Vec<std::size_t>({4});
      tensor_view.data = rust::Slice<std::uint8_t const>(
          reinterpret_cast<const std::uint8_t*>(tensor2_data.data()),
          tensor2_data.size() * sizeof(int32_t));
    }
    tensor_view.data_len = tensor_view.data.size();
    tensors.emplace_back(safetensors::PairStrTensorView{key, tensor_view});
  }

  safetensors::serialize_to_file(tensors, metadata, filename);

  // Serialize header to JSON string
  std::string header_str = header.dump();
  std::uint64_t header_size = header_str.size();

  // Calculate total data size
  std::size_t total_data_size = tensor1_data.size() * sizeof(float) +
                                tensor2_data.size() * sizeof(int32_t);

  std::cout << "Created test safetensors file: " << filename << std::endl;
  std::cout << "Header size: " << header_size << " bytes" << std::endl;
  std::cout << "Data size: " << total_data_size << " bytes" << std::endl;
  std::cout << "Total file size: " << (8 + header_size + total_data_size)
            << " bytes" << std::endl;
}

int main(int argc, char* const argv[]) {
  std::string filename = "test_model.safetensors";

  if (argc > 1) {
    filename = argv[1];
  }

  try {
    std::cout << "=== Creating Test SafeTensors File ===" << std::endl;
    create_test_safetensors(filename);
    std::cout << "\nNow you can test it with:" << std::endl;
    std::cout << "./safe_open_example " << filename << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
