cmake_minimum_required(VERSION 3.18)
project(motion_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(motion_cuda SHARED motion_field_cuda.cpp)

set_target_properties(motion_cuda PROPERTIES
    OUTPUT_NAME "motion_cuda"
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(OpenCV REQUIRED)

target_include_directories(motion_cuda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(motion_cuda PRIVATE ${OpenCV_LIBS})