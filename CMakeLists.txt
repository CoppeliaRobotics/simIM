cmake_minimum_required(VERSION 2.8.11)
project(v_repExtImage)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_MACOSX_RPATH 1)

find_package(Boost COMPONENTS thread REQUIRED)
find_package(OpenCV COMPONENTS core imgproc highgui REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(VREP REQUIRED)
#find_package(SIMD REQUIRED)

include_directories(${VREP_INCLUDE})
include_directories(${CMAKE_BINARY_DIR}/generated)
include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${SIMD_INCLUDE})

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/generated/stubs.cpp ${CMAKE_BINARY_DIR}/generated/stubs.h ${CMAKE_BINARY_DIR}/generated/lua_calltips.cpp
    COMMAND python ${CMAKE_SOURCE_DIR}/external/v_repStubsGen/generate.py --xml-file ${CMAKE_SOURCE_DIR}/callbacks.xml --lua-file ${CMAKE_SOURCE_DIR}/simExtImage.lua --gen-all ${CMAKE_BINARY_DIR}/generated
    DEPENDS ${CMAKE_SOURCE_DIR}/callbacks.xml)

set(SOURCES plugin.cpp)
add_library(v_repExtImage SHARED ${SOURCES} "${CMAKE_BINARY_DIR}/generated/stubs.cpp" "${VREP_COMMON}/v_repLib.cpp" "${CMAKE_SOURCE_DIR}/external/v_repPlusPlus/Plugin.cpp")
#target_link_libraries(v_repExtImage ${Boost_LIBRARIES} ${OpenCV_LIBS} ${VREP_LIBRARIES} ${SIMD_LIBRARIES})
target_link_libraries(v_repExtImage ${Boost_LIBRARIES} ${OpenCV_LIBS} ${VREP_LIBRARIES})
