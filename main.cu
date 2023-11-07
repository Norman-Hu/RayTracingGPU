#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Matrix.h>
#include <Camera.h>
#include <iostream>
#include <format>


__global__ void computeRays(unsigned int w, unsigned int h, float camNear, float * mat, float * out)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int totalWidth = w*3;
	float u = (float)x / (float)w;
	float v = (float)y / (float)h;
	float vec[4];
	vec[0] = 2.0f*u - 1.0f;
	vec[1] = -(2.f*v - 1.f);
	vec[2] = camNear;
	vec[3] = 1.0f;
	float newVec[4];
	if (x<w && y<h)
	{
		multVec4Mat4(vec, mat, newVec);
		out[y*totalWidth+3*x]=newVec[0]/newVec[3];
		out[y*totalWidth+3*x+1]=newVec[1]/newVec[3];
		out[y*totalWidth+3*x+2]=newVec[2]/newVec[3];
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

	Matrix4x4 proj = perspective(45.0f, 800.0f/600.0f, 0.1f, 50.0f);
	Matrix4x4 invProj;
	invertMatrix(proj, invProj);

	glViewport(0, 0, 800, 600);

	float * d_invViewProj = nullptr;
	cudaMalloc(&d_invViewProj, 16 * sizeof(float));

	dim3 blockDimensions(16, 16);
	dim3 gridDimensions((800+blockDimensions.x-1) / blockDimensions.x, (600+blockDimensions.y-1) / blockDimensions.y);

	Vec3 * h_rayArray = new Vec3[800*600];
	float * d_rayArray = nullptr;
	cudaMalloc(&d_rayArray, 3*800*600*sizeof(float));

	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);

		Matrix4x4 view = camera.GetViewMatrix();
		Matrix4x4 invView;
		invertMatrix(view, invView);

		Matrix4x4 invViewProj;
		multMat4Mat4(invView.data(), invProj.data(), invViewProj.data());

		cudaMemcpy(d_invViewProj, invViewProj.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
		computeRays<<<gridDimensions, blockDimensions>>>(800, 600, 0.1f, d_invViewProj, d_rayArray);
		cudaMemcpy(h_rayArray, d_rayArray, 3*800*600*sizeof(float), cudaMemcpyDeviceToHost);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete [] h_rayArray;
	cudaFree(d_rayArray);
	cudaFree(d_invViewProj);

	glfwTerminate();
	return 0;
}