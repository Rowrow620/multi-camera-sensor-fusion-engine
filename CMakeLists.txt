cmake_minimum_required(VERSION 3.16)
project(multi_camera_sensor_fusion_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(stereo_roi_finder stereo_roi_finder.cpp)

# Link OpenCV libraries
target_link_libraries(stereo_roi_finder PRIVATE ${OpenCV_LIBS})

# Print configuration status
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
