cmake_minimum_required(VERSION 3.23)

project(
    safetensors_cpp
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "Simple, safe way to store and distribute tensors"
)
set(SAFETENSORS_PROJECT_NAME ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 20)

option(BUILD_SHARED_LIBS "" ON)
option(SAFETENSORS_BUILD_EXAMPLES "Build examples" OFF)
option(SAFETENSORS_BUILD_BENCH "Build benchmarks" OFF)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG master
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH Cargo.toml PROFILE release)
corrosion_add_cxxbridge(safetensors_abi CRATE safetensors_rust MANIFEST_PATH src FILES lib.rs)

# Work around corrosion interface sources issue
set_target_properties(safetensors_abi PROPERTIES INTERFACE_SOURCES "")
target_include_directories(
    safetensors_abi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/safetensors_abi/include>
        $<INSTALL_INTERFACE:include>
)

option(ENABLE_LIO "Enable cross language linking time optimization" ON)
if(ENABLE_LIO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    if(ipo_supported)
        message(STATUS "Link Time Optimization (LTO) is enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        corrosion_add_target_rustflags(safetensors_rust "-Clto -C embed-bitcode=yes")
    else()
        message(
            WARNING
            "Link Time Optimization (LTO) is not supported: ${error}"
        )
    endif()
endif()

# find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)

add_library(${PROJECT_NAME} src/mmap.cpp)
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC safetensors_abi fmt::fmt
)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Add examples subdirectory
if(SAFETENSORS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SAFETENSORS_BUILD_BENCH)
    add_subdirectory(benchmark)
endif()

corrosion_install(TARGETS safetensors_rust EXPORT ${PROJECT_NAME}Targets)

# Install headers
install(DIRECTORY include/safetensors DESTINATION include)
install(
    DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/corrosion_generated/cxxbridge/safetensors_abi/include/
    DESTINATION include
)

install(
    TARGETS ${PROJECT_NAME} safetensors_abi
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Create cmake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Create config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    @ONLY
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        ${CMAKE_BINARY_DIR}/corrosion/${PROJECT_NAME}TargetsCorrosion.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)
