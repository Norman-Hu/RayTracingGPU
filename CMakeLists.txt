cmake_minimum_required(VERSION 3.24)
project(raytracer2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


add_subdirectory(third_party)

set(SOURCES main.cu
			include/CudaHelpers.h
			Vec3.cu include/Vec3.cuh
			Vec4.cu include/Vec4.cuh
			include/Ray.cuh
			Matrix.cu include/Matrix.cuh
			include/Radians.h
			include/Camera.h
			include/Material.cuh
			Object.cu include/Object.cuh
			Scene.cu include/Scene.cuh
			Rendering.cu include/Rendering.cuh
)

add_executable(raytracer2 ${SOURCES})
set_target_properties(raytracer2 PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_ARCHITECTURES 75
)
if (MSVC)
	target_compile_definitions(raytracer2 PRIVATE _USE_MATH_DEFINES)
endif()

target_link_libraries(raytracer2 PRIVATE glad glfw)
target_include_directories(raytracer2 PRIVATE include)
