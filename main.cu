#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Matrix.h>
#include <Camera.h>
#include <iostream>

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

	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);

		Matrix4x4 view = camera.GetViewMatrix();
		Matrix4x4 invView;
		invertMatrix(view, invView);

		Matrix4x4 mat;
		multMat4Vec4(invView.data(), invProj.data(), mat.data());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}