cmake_minimum_required(VERSION 3.23)

list(APPEND CMAKE_PREFIX_PATH)

# Function to find PyTorch from Python environment
function(find_pytorch_from_python)
    # Find Python executable
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Get PyTorch installation path from Python
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_PYTHON_RESULT
    )

    if(TORCH_PYTHON_RESULT EQUAL 0)
        message(
            STATUS
            "Found PyTorch cmake path from Python: ${TORCH_CMAKE_PREFIX_PATH}"
        )
        set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH} PARENT_SCOPE)
    else()
        # Fallback: try to get torch installation directory
        execute_process(
            COMMAND
                ${Python3_EXECUTABLE} -c
                "import torch; import os; print(os.path.dirname(torch.__file__))"
            OUTPUT_VARIABLE TORCH_INSTALL_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE TORCH_PATH_RESULT
        )

        if(TORCH_PATH_RESULT EQUAL 0)
            message(
                STATUS
                "Found PyTorch installation path: ${TORCH_INSTALL_PATH}"
            )
            set(CMAKE_PREFIX_PATH
                "${TORCH_INSTALL_PATH}/share/cmake"
                PARENT_SCOPE
            )
        else()
            message(WARNING "Could not find PyTorch from Python environment")
        endif()
    endif()
endfunction()

find_package(CUDAToolkit REQUIRED)

find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(STATUS "PyTorch NVTX headers workaround: Yes")
    # only do this if nvToolsExt is not defined and CUDA::nvtx3 exists
    if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        # ensure that PyTorch is told to use NVTX3 headers
        target_compile_definitions(CUDA::nvToolsExt INTERFACE TORCH_CUDA_USE_NVTX3)
        target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
    endif()
    message(STATUS "Find Torch from python environment")
    find_pytorch_from_python()
    find_package(Torch CONFIG)
    if(NOT Torch_FOUND)
        message(FATAL_ERROR "pip install torch")
    endif()
endif()

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

add_executable(bench_cpp bench.cpp)
target_link_libraries(bench_cpp PRIVATE safetensors_cpp ${TORCH_LIBRARIES})

# Add optimized benchmark
add_executable(bench_optimized bench_optimized.cpp)
target_link_libraries(bench_optimized PRIVATE safetensors_cpp ${TORCH_LIBRARIES})

add_custom_target(
    bench_py
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/bench.py
        ${CMAKE_CURRENT_BINARY_DIR}/bench.py
    COMMENT "Copying benchmark files to build directory"
)

add_dependencies(bench_cpp bench_py)
