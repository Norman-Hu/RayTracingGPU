cmake_minimum_required(VERSION 3.16)
project(raytracer2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


add_subdirectory(third_party)

set(SOURCES main.cu
			include/Vec3.h
			include/Ray.h
			include/Matrix.h
			include/Radians.h
			include/Camera.h
			include/Object.h
			Scene.cpp include/Scene.h
)

add_executable(raytracer2 ${SOURCES})
set_target_properties(raytracer2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raytracer2 PRIVATE glad glfw)
target_include_directories(raytracer2 PRIVATE include)
