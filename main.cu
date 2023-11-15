#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Matrix.cuh>
#include <Camera.h>
#include <Scene.cuh>
#include <iostream>
#include <format>


__global__ void initScene(Scene ** ptrScene)
{
	*ptrScene = new Scene();
	(*ptrScene)->size = 1;
	(*ptrScene)->objectList = new Hitable*[1];
	(*ptrScene)->objectList[0] = new Sphere();
	static_cast<Sphere*>((*ptrScene)->objectList[0])->c = {5.0f, 0.0f, 0.0f};
	static_cast<Sphere*>((*ptrScene)->objectList[0])->r = 1.0f;
}

__global__ void deleteScene(Scene * ptrScene)
{
	delete ptrScene;
}

__global__ void computeRays(unsigned int w, unsigned int h, float camNear, Matrix4x4 invViewProj, Vec3 * out)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	float u = (float)x / (float)w;
	float v = (float)y / (float)h;
	Vec4 vec{2.0f*u - 1.0f, -(2.f*v - 1.f), camNear, 1.0f};
	if (x<w && y<h)
	{
		vec = vec * invViewProj;
		out[y*w+x] = Vec3{vec.x, vec.y, vec.z}/vec.w;
	}
}

int main(int argc, char **argv)
{
	GLFWwindow * window;
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


	window = glfwCreateWindow(800, 600, "Hello World", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	Camera camera;

	Matrix4x4 proj = Matrix4x4::perspective(45.0f, 800.0f/600.0f, 0.1f, 50.0f);
	Matrix4x4 invProj;
	Matrix4x4::invertMatrix(proj, invProj);

	glViewport(0, 0, 800, 600);

	dim3 blockDimensions(16, 16);
	dim3 gridDimensions((800+blockDimensions.x-1) / blockDimensions.x, (600+blockDimensions.y-1) / blockDimensions.y);

	Vec3 * h_rayArray = new Vec3[800*600];
	Vec3 * d_rayArray = nullptr;
	cudaMalloc(&d_rayArray, 800*600*sizeof(Vec3));

	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);

		Matrix4x4 view = camera.GetViewMatrix();
		Matrix4x4 invView;
		Matrix4x4::invertMatrix(view, invView);

		Matrix4x4 invViewProj = invView * invProj;

		computeRays<<<gridDimensions, blockDimensions>>>(800, 600, 0.1f, invViewProj, d_rayArray);
		cudaMemcpy(h_rayArray, d_rayArray, 800*600*sizeof(Vec3), cudaMemcpyDeviceToHost);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete [] h_rayArray;
	cudaFree(d_rayArray);

	glfwTerminate();
	return 0;
}