cmake_minimum_required (VERSION 3.0)
project (myproject LANGUAGES CUDA CXX)

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
aux_source_directory(. DIR_SRCS)
add_subdirectory(src)

add_library(gpu_conv_lib STATIC gpu_layer/gpu_conv.cu)

add_executable(demo ${DIR_SRCS})
target_link_libraries(demo gpu_conv_lib MiniDNNLib)