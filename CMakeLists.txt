cmake_minimum_required(VERSION 2.8.11)
project(v_repExtImage)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

find_package(Boost COMPONENTS thread REQUIRED)
find_package(OpenCV COMPONENTS core imgproc highgui REQUIRED)

if(${OpenCV_VERSION} VERSION_GREATER 3.0.0)
    # since opencv3, imread is in the imgcodecs module, VideoCapture in videoio
    find_package(OpenCV COMPONENTS imgcodecs videoio REQUIRED)
endif()

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CoppeliaRobotics-modules)
find_package(VREP 3.5.0 REQUIRED)
#find_package(SIMD REQUIRED)

include_directories(${VREP_INCLUDE})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${SIMD_INCLUDE})

vrep_generate_stubs(${CMAKE_CURRENT_BINARY_DIR}/generated XML_FILE ${CMAKE_CURRENT_SOURCE_DIR}/callbacks.xml LUA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/simExtImage.lua)

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

set(SOURCES plugin.cpp)
add_library(v_repExtImage SHARED ${SOURCES} "${CMAKE_CURRENT_BINARY_DIR}/generated/stubs.cpp" "${VREP_EXPORTED_SOURCES}" "${CMAKE_CURRENT_SOURCE_DIR}/external/v_repPlusPlus/Plugin.cpp")
#target_link_libraries(v_repExtImage ${Boost_LIBRARIES} ${OpenCV_LIBS} ${VREP_LIBRARIES} ${SIMD_LIBRARIES})
target_link_libraries(v_repExtImage ${Boost_LIBRARIES} ${OpenCV_LIBS} ${VREP_LIBRARIES})

install(TARGETS v_repExtImage DESTINATION ${VREP_ROOT})

