cmake_minimum_required(VERSION 3.18)
project(MNISTClassifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/opt/libtorch")
set(CMAKE_CUDA_ARCHITECTURES "89")

# CUDA::nvToolsExt alias for LibTorch compatibility
find_library(NVTOOLSEXT_LIBRARY NAMES nvToolsExt PATHS "/usr/local/cuda-12.9/lib64")
if (NVTOOLSEXT_LIBRARY)
  add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
  set_target_properties(CUDA::nvToolsExt PROPERTIES
    IMPORTED_LOCATION ${NVTOOLSEXT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/cuda-12.9/include")
endif()

find_package(Torch REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(mnist_classifier
  src/main.cpp
  src/net.cpp
)

target_link_libraries(mnist_classifier "${TORCH_LIBRARIES}")
set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)
target_link_libraries(mnist_classifier "${TORCH_LIBRARIES}")


# Install rules
install(TARGETS mnist_classifier DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY models/ DESTINATION share/mnist_classifier/models OPTIONAL)
include_directories(${CMAKE_SOURCE_DIR}/src/include)
