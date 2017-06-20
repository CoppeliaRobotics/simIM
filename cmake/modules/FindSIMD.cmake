cmake_minimum_required(VERSION 2.8.12)

if(NOT SIMD_DIR AND DEFINED ENV{SIMD_DIR})
    set(SIMD_DIR "$ENV{SIMD_DIR}")
endif()

if(NOT SIMD_DIR)
    message(FATAL_ERROR "Please set SIMD_DIR to the root of the Simd source tree.")
endif()

file(TO_CMAKE_PATH "${SIMD_DIR}" SIMD_DIR)

if(EXISTS "${SIMD_DIR}/prj/cmake/libSimd.a")
    set(SIMD_FOUND TRUE)
    if(NOT SIMD_FIND_QUIETLY)
        message(STATUS "Found Simd library at ${SIMD_DIR}/prj/cmake/libSimd.a")
    endif()
    set(SIMD_INCLUDE "${SIMD_DIR}/src")
    set(SIMD_LIBRARIES "${SIMD_DIR}/prj/cmake/libSimd.a")
else()
    if(SIMD_FIND_REQUIRED)
        message(FATAL_ERROR "The specified SIMD_DIR does not contain the built library.")
    endif()
endif()

