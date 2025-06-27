# SafeTensors C++ Examples

This directory contains examples demonstrating how to use the SafeTensors C++ library.

## Examples

### 1. safe_open_example.cpp
Demonstrates how to use the `SafeOpen` class to load and inspect safetensors files:
- Load a safetensors file
- List all tensor keys
- Access tensor metadata
- Print tensor information (shape, dtype, data)
- Access specific tensors by name

### 2. create_test_data.cpp
Creates a simple test safetensors file that can be used with the other examples:
- Creates tensors with different shapes and data types
- Adds metadata to the file
- Writes a properly formatted safetensors file

## Building the Examples

The examples are automatically built when you build the main safetensors_cpp project with examples enabled.

### From the build directory:
```bash
cd /path/to/safetensors/bindings/cpp/build
make
```

### Or build just the examples:
```bash
cd /path/to/safetensors/bindings/cpp/build
make safe_open_example create_test_data
```

## Running the Examples

### 1. Create test data:
```bash
./create_test_data [optional_filename.safetensors]
```
This creates a test safetensors file (default: `test_model.safetensors`)

### 2. Inspect a safetensors file:
```bash
./safe_open_example <safetensors_file>
```

### Complete example workflow:
```bash
# Create test data
./create_test_data my_test.safetensors

# Inspect the created file
./safe_open_example my_test.safetensors
```

## Expected Output

The `safe_open_example` will display:
- File metadata
- List of all tensors with their properties:
  - Shape (dimensions)
  - Data type (F32, I32, etc.)
  - Total number of elements
  - Data size in bytes
  - Sample of the actual data values

## Integration with Your Project

To use SafeTensors in your own C++ project:

1. Link against the safetensors_cpp library
2. Include the header: `#include <safetensors/safetensors.hpp>`
3. Use the `SafeOpen` class as demonstrated in the examples

Example CMakeLists.txt for your project:
```cmake
find_package(safetensors_cpp REQUIRED)
target_link_libraries(your_target PRIVATE safetensors_cpp::safetensors_cpp)
```
