#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Matrix.cuh>
#include <Camera.h>
#include <Scene.cuh>
#include <iostream>
#include <format>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <surface_indirect_functions.h>
#include <surface_functions.h>


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

__global__ void testFillFramebuffer(unsigned int w, unsigned int h, cudaSurfaceObject_t surface)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x<w && y<h)
	{
		uchar4 val = {255, 255, 0, 255};
		surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
	}
}

void destroyBuffers(unsigned int rb, unsigned int fb)
{
	glDeleteRenderbuffers(1, &rb);
	glDeleteFramebuffers(1, &fb);
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

	// Create buffers for rendering
	unsigned int rb, fb;
	glGenRenderbuffers(1, &rb);
	glBindRenderbuffer(GL_RENDERBUFFER, rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 800, 600);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glGenFramebuffers(1, &fb);

	glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	cudaGraphicsResource_t gr;
	cudaGraphicsGLRegisterImage(&gr, rb, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
	// get cuda array : map + array references + unmap
	cudaArray_t ar;
	cudaGraphicsMapResources(1, &gr);
	cudaGraphicsSubResourceGetMappedArray(&ar, gr, 0, 0);
	cudaGraphicsUnmapResources(1, &gr);


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

		// map cuda array
		cudaGraphicsMapResources(1, &gr);

		cudaResourceDesc resDesc{ .resType = cudaResourceTypeArray };
		resDesc.res.array.array = ar;
		cudaSurfaceObject_t surfObj;
		cudaCreateSurfaceObject(&surfObj, &resDesc);

		Matrix4x4 view = camera.GetViewMatrix();
		Matrix4x4 invView;
		Matrix4x4::invertMatrix(view, invView);

		Matrix4x4 invViewProj = invView * invProj;

		computeRays<<<gridDimensions, blockDimensions>>>(800, 600, 0.1f, invViewProj, d_rayArray);
		cudaMemcpy(h_rayArray, d_rayArray, 800*600*sizeof(Vec3), cudaMemcpyDeviceToHost);

		testFillFramebuffer<<<gridDimensions, blockDimensions>>>(800, 600, surfObj);

		// unmap cuda array
		cudaGraphicsUnmapResources(1, &gr);



		glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, 800, 600, 0, 0, 800, 600, GL_COLOR_BUFFER_BIT, GL_LINEAR);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete [] h_rayArray;
	cudaFree(d_rayArray);

	// cleanup
	destroyBuffers(rb, fb);

	glfwTerminate();
	return 0;
}