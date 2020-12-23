cmake_minimum_required(VERSION 3.10)
project(tensorflow)

set(CMAKE_CXX_STANDARD 14)


find_library(TENSORFLOW_LIB tensorflow)

add_executable(tensorflow main.cpp)

target_link_libraries(tensorflow "${TENSORFLOW_LIB}")