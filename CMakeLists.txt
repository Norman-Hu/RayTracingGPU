cmake_minimum_required(VERSION 3.22)
project(raytracer2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


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
			InputHandler.cu include/InputHandler.cuh
			Light.cu include/Light.cuh
			Importer.cu include/Importer.cuh
)

add_executable(raytracer2 ${SOURCES})
set_target_properties(raytracer2 PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_ARCHITECTURES 75
)
if (MSVC)
	target_compile_definitions(raytracer2 PRIVATE _USE_MATH_DEFINES)
endif()
target_compile_options(raytracer2 PRIVATE -Xcudafe --diag_suppress=20012 -G)

target_link_libraries(raytracer2 PRIVATE glad glfw assimp)
target_include_directories(raytracer2 PRIVATE include)
