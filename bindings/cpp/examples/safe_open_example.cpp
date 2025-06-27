/*
 * Copyright (c) 2025 Dapeng Feng
 * All rights reserved.
 */

// SafeTensors C++ Example: SafeOpen Usage
// This example demonstrates how to use the SafeOpen class to:
// 1. Load a safetensors file
// 2. List all tensor keys
// 3. Access tensor data and metadata
// 4. Print tensor information

#include <iomanip>
#include <iostream>

#include "safetensors/safetensors.hpp"

// Helper function to get dtype name as string
std::string dtype_to_string(safetensors::Dtype dtype) {
  switch (dtype) {
    case safetensors::Dtype::F64:
      return "F64";
    case safetensors::Dtype::F32:
      return "F32";
    case safetensors::Dtype::F16:
      return "F16";
    case safetensors::Dtype::BF16:
      return "BF16";
    case safetensors::Dtype::I64:
      return "I64";
    case safetensors::Dtype::I32:
      return "I32";
    case safetensors::Dtype::I16:
      return "I16";
    case safetensors::Dtype::I8:
      return "I8";
    case safetensors::Dtype::U64:
      return "U64";
    case safetensors::Dtype::U32:
      return "U32";
    case safetensors::Dtype::U16:
      return "U16";
    case safetensors::Dtype::U8:
      return "U8";
    case safetensors::Dtype::BOOL:
      return "BOOL";
    default:
      return "UNKNOWN";
  }
}

// Helper function to calculate total elements in tensor
std::size_t calculate_total_elements(const std::vector<std::size_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1UL,
                         std::multiplies<std::size_t>());
}

// Helper function to print tensor shape
void print_shape(const std::vector<std::size_t>& shape) {
  std::cout << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << shape[i];
  }
  std::cout << "]";
}

// Helper function to print some tensor data (first few elements)
void print_tensor_data(const safetensors::SafeOpen::TensorView& tensor,
                       std::size_t max_elements = 10) {
  std::size_t total_elements = calculate_total_elements(tensor.shape);
  std::size_t elements_to_print = std::min(max_elements, total_elements);

  std::cout << "    Data (first " << elements_to_print << " elements): ";

  switch (tensor.dtype) {
    case safetensors::Dtype::F32: {
      const float* data = static_cast<const float*>(tensor.data_ptr);
      for (std::size_t i = 0; i < elements_to_print; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << data[i];
      }
      break;
    }
    case safetensors::Dtype::F64: {
      const double* data = static_cast<const double*>(tensor.data_ptr);
      for (std::size_t i = 0; i < elements_to_print; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << data[i];
      }
      break;
    }
    case safetensors::Dtype::I32: {
      const int32_t* data = static_cast<const int32_t*>(tensor.data_ptr);
      for (std::size_t i = 0; i < elements_to_print; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
      }
      break;
    }
    case safetensors::Dtype::I64: {
      const int64_t* data = static_cast<const int64_t*>(tensor.data_ptr);
      for (std::size_t i = 0; i < elements_to_print; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
      }
      break;
    }
    case safetensors::Dtype::U8: {
      const uint8_t* data = static_cast<const uint8_t*>(tensor.data_ptr);
      for (std::size_t i = 0; i < elements_to_print; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << static_cast<int>(data[i]);
      }
      break;
    }
    default:
      std::cout << "[binary data not displayed for this dtype]";
      break;
  }

  if (total_elements > elements_to_print) {
    std::cout << "... (" << (total_elements - elements_to_print)
              << " more elements)";
  }
  std::cout << std::endl;
}

int main(int argc, char* const argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <safetensors_file>" << std::endl;
    std::cerr << "Example: " << argv[0] << " model.safetensors" << std::endl;
    return 1;
  }

  const std::string filename = argv[1];

  try {
    std::cout << "=== SafeTensors C++ Example ===" << std::endl;
    std::cout << "Loading file: " << filename << std::endl << std::endl;

    // Create SafeOpen instance
    safetensors::SafeOpen safe_open(filename);

    // Get and display metadata
    std::cout << "=== Metadata ===" << std::endl;
    auto metadata = safe_open.get_metadata();
    if (metadata.empty()) {
      std::cout << "No metadata found in the file." << std::endl;
    } else {
      // cppcheck-suppress unassignedVariable
      for (const auto& [key, value] : metadata) {
        std::cout << "  " << key << ": " << value << std::endl;
      }
    }
    std::cout << std::endl;

    // Get all tensor keys
    auto tensor_keys = safe_open.keys();
    std::cout << "=== Tensors ===" << std::endl;
    std::cout << "Found " << tensor_keys.size() << " tensor(s):" << std::endl
              << std::endl;

    // Iterate through all tensors and display information
    for (const auto& key : tensor_keys) {
      std::cout << "Tensor: \"" << key << "\"" << std::endl;

      try {
        // Get tensor view
        auto tensor = safe_open.get_tensor(key);

        // Display tensor information
        std::cout << "    Shape: ";
        print_shape(tensor.shape);
        std::cout << std::endl;

        std::cout << "    Dtype: " << dtype_to_string(tensor.dtype)
                  << std::endl;

        std::cout << "    Total elements: "
                  << calculate_total_elements(tensor.shape) << std::endl;

        std::cout << "    Data size: " << tensor.data_len << " bytes"
                  << std::endl;

        // Print some sample data
        print_tensor_data(tensor);
      } catch (const std::exception& e) {
        std::cerr << "    Error loading tensor: " << e.what() << std::endl;
      }

      std::cout << std::endl;
    }

    // Example: Access a specific tensor by name (if it exists)
    if (!tensor_keys.empty()) {
      const std::string first_key = tensor_keys[0];
      std::cout << "=== Accessing Specific Tensor ===" << std::endl;
      std::cout << "Accessing tensor: \"" << first_key << "\"" << std::endl;

      auto tensor = safe_open.get_tensor(first_key);
      std::cout << "Successfully accessed tensor with shape: ";
      print_shape(tensor.shape);
      std::cout << std::endl;
    }

    std::cout << "=== Example completed successfully ===" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
