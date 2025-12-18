cmake_minimum_required(VERSION 3.20)

project(fast_mnist_nn VERSION 0.1.0 LANGUAGES CXX)

option(FAST_MNIST_ENABLE_OPENMP "Enable OpenMP support" ON)
option(FAST_MNIST_ENABLE_NATIVE "Enable -march=native" OFF)

add_library(fast_mnist
  src/Matrix.cpp
  src/NeuralNet.cpp
)

target_include_directories(fast_mnist
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(fast_mnist PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(fast_mnist PRIVATE /W4)
else()
  target_compile_options(fast_mnist PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(FAST_MNIST_ENABLE_NATIVE)
  if(MSVC)
    message(WARNING "FAST_MNIST_ENABLE_NATIVE is not supported on MSVC")
  else()
    target_compile_options(fast_mnist PRIVATE -march=native)
  endif()
endif()

if(FAST_MNIST_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_mnist PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fast_mnist PUBLIC FAST_MNIST_USE_OPENMP=1)
  else()
    message(STATUS "OpenMP not found; building without OpenMP")
  endif()
endif()

add_executable(fast_mnist_cli apps/fast_mnist_cli.cpp)
target_link_libraries(fast_mnist_cli PRIVATE fast_mnist)

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(fast_mnist_tests
    tests/test_matrix.cpp
  )
  target_link_libraries(fast_mnist_tests
    PRIVATE
      fast_mnist
      Catch2::Catch2WithMain
  )

  include(Catch)
  catch_discover_tests(fast_mnist_tests)
endif()
