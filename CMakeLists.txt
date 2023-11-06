cmake_minimum_required(VERSION 3.16)
project(raytracer2)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(third_party)

add_executable(raytracer2 main.cpp)
target_link_libraries(raytracer2 PRIVATE glad glfw)