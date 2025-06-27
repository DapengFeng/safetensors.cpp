cmake_minimum_required(VERSION 3.21)

# Add the safe_open_example executable
add_executable(safe_open_example safe_open_example.cpp)
target_link_libraries(safe_open_example PRIVATE safetensors_cpp)

# Add the create_test_data executable
add_executable(create_test_data create_test_data.cpp)
target_link_libraries(create_test_data PRIVATE safetensors_cpp)
