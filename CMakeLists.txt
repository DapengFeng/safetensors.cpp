cmake_minimum_required(VERSION 3.23)

project(safetensors)

option(SAFETENSOR_C "Build c bindings" OFF)
option(SAFETENSOR_CPP "Build cpp bindings" ON)

if(SAFETENSOR_C)
    add_subdirectory(bindings/c)
endif()

if(SAFETENSOR_CPP)
    add_subdirectory(bindings/cpp)
endif()
