cmake_minimum_required(VERSION 3.21)

project(
    safetensors_cpp
    VERSION 0.3.0
    LANGUAGES C CXX
    DESCRIPTION "Simple, safe way to store and distribute tensors"
)
set(SAFETENSORS_PROJECT_NAME ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 20)

option(BUILD_SHARED_LIBS "" ON)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH Cargo.toml PROFILE release)
corrosion_add_cxxbridge(safetensors_abi CRATE safetensors_rust MANIFEST_PATH src FILES lib.rs)

option(ENABLE_LIO "Enable cross language linking time optimization" ON)
if(ENABLE_LIO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    if(ipo_supported)
        message(STATUS "Link Time Optimization (LTO) is enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        # corrosion_add_target_rustflags(safetensors_rust "-Clto")
    else()
        message(
            WARNING
            "Link Time Optimization (LTO) is not supported: ${error}"
        )
    endif()
endif()

find_package(nlohmann_json REQUIRED)

add_library(${PROJECT_NAME} src/mmap.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC safetensors_abi nlohmann_json::nlohmann_json)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/safetensors DESTINATION include)
# # install(EXPORT ${PROJECT_NAME}-targets
# #     FILE ${PROJECT_NAME}-targets.cmake
# #     NAMESPACE ${PROJECT_NAME}::
# #     DESTINATION lib/cmake/${PROJECT_NAME}
# # )
# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )
# # configure_package_config_file(
# #     "${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}-config.cmake.in"
# #     "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
# #     INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
# #     NO_SET_AND_CHECK_MACRO
# #     NO_CHECK_REQUIRED_COMPONENTS_MACRO
# # )
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
#     DESTINATION lib/cmake/${PROJECT_NAME}
# )