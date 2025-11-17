cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(dice-reader)

find_package(OpenCV REQUIRED)

add_library(dbscan
    external/dbscan/dbscan.cpp
)

target_include_directories(dbscan PRIVATE external/dbscan/vendor)

add_executable(dice-reader 
    src/main.cpp
)

target_include_directories(dice-reader PRIVATE ${OpenCV_INCLUDE_DIRS} external/dbscan)

target_link_libraries(dice-reader PRIVATE ${OpenCV_LIBS} dbscan)
