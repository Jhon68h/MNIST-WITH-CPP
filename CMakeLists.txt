cmake_minimum_required(VERSION 3.10)
project(MNIST)

# Encontrar OpenCV y ROOT
find_package(OpenCV REQUIRED)
#find_package(ROOT REQUIRED)

# Configurar las variables de entorno de ROOT
#include(${ROOT_USE_FILE})

# Incluir directorios de cabecera
include_directories(${OpenCV_INCLUDE_DIRS} /usr/include/opencv4)
# Configurar VDT
#set(VDT_INCLUDE_DIR /usr/local/include)
#set(VDT_LIBRARY /usr/local/lib/libvdt.so)

# Añadir el ejecutable
add_executable(${PROJECT_NAME} main.cpp Neuron.cpp)

# Enlazar las bibliotecas necesarias
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
