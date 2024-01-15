#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Matrix.cuh>
#include <Camera.h>
#include <Scene.cuh>
#include <Rendering.cuh>
#include <iostream>
#include <InputHandler.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <CudaHelpers.h>
#include <curand.h>
#include <curand_kernel.h>
#include <Importer.cuh>


void destroyBuffers(unsigned int rb, unsigned int fb)
{
	glDeleteRenderbuffers(1, &rb);
	glDeleteFramebuffers(1, &fb);
}

// from https://gist.github.com/allanmac/4ff11985c3562830989f
void setTitleFPS(GLFWwindow * pWindow)
{
	static float previousStamp = 0.0f;
	static int count = 0;

	float currentStamp = glfwGetTime();
	float elapsed = currentStamp - previousStamp;

	if (elapsed > 0.5f)
	{
		previousStamp = currentStamp;
		float fps = count / elapsed;
		int w, h;
		glfwGetFramebufferSize(pWindow,&w,&h);
		char tmp[64];
      	sprintf(tmp,"(%u x %u) - FPS: %.2f",w,h,fps);
		glfwSetWindowTitle(pWindow, tmp);
		count = 0;
	}
	count++;
}

int main(int argc, char **argv)
{
    cudaDeviceSetLimit(cudaLimitStackSize, 2048);

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
	glfwSwapInterval(0); // disable vsync

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
	Matrix4x4 proj = Matrix4x4::perspective(radians(120), 800.0f/600.0f, 0.1f, 50.0f);
	Matrix4x4 invProj;
	Matrix4x4::invertMatrix(proj, invProj);

	glViewport(0, 0, 800, 600);

	// setup glfw callbacks
	InputHandler handler(window, &camera);
	glfwSetWindowUserPointer(window, &handler);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	// dimensions of the render surface
	dim3 blockDimensions(16, 16);
	dim3 gridDimensions((800+blockDimensions.x-1) / blockDimensions.x, (600+blockDimensions.y-1) / blockDimensions.y);
	// setup gpu memory

	int threadCount = gridDimensions.x*blockDimensions.x*gridDimensions.y*blockDimensions.y;
	Scene * d_scene = importSceneToGPU("scenes/cornell-light-lowpoly2.glb");
	syncAndCheckErrors();

	curandState_t * randState;
	cudaMalloc(&randState, threadCount * sizeof(curandState_t));
	setupRandomState<<<gridDimensions, blockDimensions>>>(randState, time(nullptr));

	while (!glfwWindowShouldClose(window))
	{
		setTitleFPS(window);

		glClear(GL_COLOR_BUFFER_BIT);

		// map cuda array
		cudaGraphicsMapResources(1, &gr);

		cudaResourceDesc resDesc;
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = ar;
		cudaSurfaceObject_t surfObj;
		cudaCreateSurfaceObject(&surfObj, &resDesc);

		Matrix4x4 rayTransform = invProj * camera.GetViewMatrix();

		render<<<gridDimensions, blockDimensions>>>(d_scene, 800, 600, 0.1f, camera.Position, rayTransform, surfObj, randState);
		syncAndCheckErrors();

		// unmap cuda array
		cudaGraphicsUnmapResources(1, &gr);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, 800, 600, 0, 600, 800, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);

		glfwSwapBuffers(window);
		glfwPollEvents();
		handler.handleInputs();
	}

	// cleanup
	destroyScene(d_scene);
	destroyBuffers(rb, fb);
	cudaFree(randState);

	glfwTerminate();
	return 0;
}