set(CMAKE_CXX_COMPILER icpx)

cmake_minimum_required (VERSION 3.4)

project(FPGAOverlay CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (operators)
add_subdirectory (examples)
