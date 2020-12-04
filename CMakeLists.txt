cmake_minimum_required(VERSION 2.8.11)
project(simExtIM)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_MACOSX_RPATH 1)

find_package(Boost COMPONENTS thread REQUIRED)
find_package(OpenCV COMPONENTS core imgproc highgui REQUIRED)

if(${OpenCV_VERSION} VERSION_GREATER 3.0.0)
    # since opencv3, imread is in the imgcodecs module, VideoCapture in videoio
    find_package(OpenCV COMPONENTS imgcodecs videoio REQUIRED)
endif()

if(NOT LIBPLUGIN_DIR)
    if(DEFINED ENV{COPPELIASIM_ROOT_DIR})
        set(LIBPLUGIN_DIR $ENV{COPPELIASIM_ROOT_DIR}/programming/libPlugin)
    else()
        set(LIBPLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libPlugin)
    endif()
endif()
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
    ${LIBPLUGIN_DIR}/cmake)
find_package(CoppeliaSim 4.1.0.0 REQUIRED)
#find_package(SIMD REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${SIMD_INCLUDE})

coppeliasim_generate_stubs(${CMAKE_CURRENT_BINARY_DIR}/generated XML_FILE ${CMAKE_CURRENT_SOURCE_DIR}/callbacks.xml LUA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/simIM.lua)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${OpenCV_LIBS})
check_cxx_source_compiles("
#include <opencv2/core/core.hpp>
int main() {
    cv::Mat im(100, 100, CV_8UC1);
    cv::arrowedLine(im, cv::Point(10, 20), cv::Point(50, 90), cv::Scalar(100));
}
" TEST_CV_ARROWEDLINE_PASSED)
if(TEST_CV_ARROWEDLINE_PASSED)
    set(HAVE_CV_ARROWEDLINE 1)
else()
    set(HAVE_CV_ARROWEDLINE 0)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

coppeliasim_add_plugin(simExtIM SOURCES plugin.cpp)
target_link_libraries(simExtIM Boost::boost Boost::thread)
target_link_libraries(simExtIM ${OpenCV_LIBS})
#target_link_libraries(simExtIM ${SIMD_LIBRARIES})
