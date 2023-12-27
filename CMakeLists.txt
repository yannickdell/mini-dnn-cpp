cmake_minimum_required (VERSION 3.0)

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
project (DNN LANGUAGES CUDA CXX)
aux_source_directory(. SRC) 
add_subdirectory(src) 
add_subdirectory (src/gpu_layer)

add_executable(demo ${DIR_SRCS})
target_link_libraries(demo 116)
target_link_libraries(demo cuda_116)