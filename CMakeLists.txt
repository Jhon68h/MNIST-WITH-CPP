cmake_minimum_required(VERSION 3.10)
project(MNIST)
find_package(OpenCV REQUIRED)
#find_package(ROOT REQUIRED)
#include(${ROOT_USE_FILE})

include_directories(${OpenCV_INCLUDE_DIRS} /usr/include/opencv4)
#set(VDT_INCLUDE_DIR /usr/local/include)
#set(VDT_LIBRARY /usr/local/lib/libvdt.so)

add_executable(${PROJECT_NAME} main.cpp Neuron.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
