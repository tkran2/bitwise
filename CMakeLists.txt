cmake_minimum_required(VERSION 3.20)
project(bitwise LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BITWISE_AVX512 "Build the AVX-512 kernels (otherwise the portable scalar path)" ON)
option(BITWISE_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

# No fast-math, and no implicit FMA contraction: every fused multiply-add in the
# kernels is written explicitly, so the compiler cannot change rounding per call site.
set(BW_FLAGS -Wall -Wextra -ffp-contract=off -fno-fast-math)
if(BITWISE_AVX512)
  list(APPEND BW_FLAGS -mavx512f -mavx512bw -mavx512vl -mfma)
endif()
if(BITWISE_SANITIZE)
  list(APPEND BW_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)
add_library(bitwise_core STATIC src/kernels.cpp src/model.cpp src/block_manager.cpp src/engine.cpp)
target_include_directories(bitwise_core PUBLIC src)
target_compile_options(bitwise_core PRIVATE ${BW_FLAGS} -fPIC)
target_link_libraries(bitwise_core PUBLIC Threads::Threads)

add_executable(bitwise_tests tests/test_main.cpp)
target_compile_options(bitwise_tests PRIVATE ${BW_FLAGS})
target_link_libraries(bitwise_tests PRIVATE bitwise_core)

add_executable(bitwise_sim sim/simulate.cpp)
target_compile_options(bitwise_sim PRIVATE ${BW_FLAGS})
target_link_libraries(bitwise_sim PRIVATE bitwise_core)

enable_testing()
add_test(NAME unit COMMAND bitwise_tests)
add_test(NAME sim_smoke COMMAND bitwise_sim --traces 200)

execute_process(COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(PYBIND11_DIR)
  set(pybind11_DIR ${PYBIND11_DIR})
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_bitwise python/bindings.cpp)
  target_compile_options(_bitwise PRIVATE ${BW_FLAGS})
  target_link_libraries(_bitwise PRIVATE bitwise_core)
  set_target_properties(_bitwise PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/bitwise)
endif()
