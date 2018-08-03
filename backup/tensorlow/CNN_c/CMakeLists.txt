cmake_minimum_required(VERSION 3.10)
project(CNN_c)

set(CMAKE_CXX_STANDARD 14)

add_library(CNN_c SHARED library.cpp library.h)