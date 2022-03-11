cmake_minimum_required (VERSION 3.16)

set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING INTERNAL FORCE )

# Default build type is Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build: Release Debug."
      FORCE)
endif()

project (subview VERSION 0.0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

set(CMAKE_CXX_STANDARD "17")

set(KOKKOS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kokkos")
set(KOKKOS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/kokkos")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
endif()

if("${Kokkos_ENABLE_OPENMP}" STREQUAL "ON")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(OPENMP_LINK_FLAGS OpenMP::OpenMP_CXX)
endif()

add_subdirectory(${KOKKOS_SOURCE_DIR})

include_directories(SYSTEM ${KOKKOS_SOURCE_DIR}/core/src)
include_directories(SYSTEM ${KOKKOS_SOURCE_DIR}/containers/src)
include_directories(SYSTEM ${KOKKOS_BINARY_DIR})

add_executable(
  subview
  main.cpp)

target_link_libraries(
  subview
  kokkos
  ${KOKKOS_CXX_FLAGS}
  ${OPENMP_LINK_FLAGS}
  stdc++fs
  )
