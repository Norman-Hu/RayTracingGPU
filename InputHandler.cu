#include <InputHandler.cuh>
#include <Camera.h>
#include <Vec3.cuh>


InputHandler::InputHandler(GLFWwindow * _pWindow, Camera * _pCamera)
: pWindow(_pWindow)
, pCamera(_pCamera)
, cursorCaptured(false)
{
	if (glfwRawMouseMotionSupported())
		glfwSetInputMode(pWindow, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
}

void InputHandler::handleInputs()
{
	float vertical = 0.0f;
	if (glfwGetKey(pWindow, GLFW_KEY_W))
		vertical += 1.f;
	if (glfwGetKey(pWindow, GLFW_KEY_S))
		vertical -= 1.f;

	float horizontal = 0.0f;
	if (glfwGetKey(pWindow, GLFW_KEY_A))
		horizontal -= 1.f;
	if (glfwGetKey(pWindow, GLFW_KEY_D))
		horizontal += 1.f;

	if (horizontal != 0.0f && vertical != 0.0f)
	{
		horizontal /= M_SQRT2;
		vertical /= M_SQRT2;
	}

	pCamera->Position += pCamera->Front*vertical + pCamera->Right*horizontal;
}

void InputHandler::keyCallback(int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE)
	{
		cursorCaptured = false;
		glfwSetInputMode(pWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void InputHandler::mouseButtonCallback(int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		cursorCaptured = true;
		glfwSetInputMode(pWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
}

void InputHandler::cursorPosCallback(double xpos, double ypos)
{
	if (!cursorCaptured)
		return;

	pCamera->ProcessMouseMovement(xpos, ypos, true);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	InputHandler * pHandler = static_cast<InputHandler *>(glfwGetWindowUserPointer(window));
	pHandler->keyCallback(key, scancode, action, mods);
}

void cursorPosCallback(GLFWwindow * window, double xpos, double ypos)
{
	InputHandler * pHandler = static_cast<InputHandler *>(glfwGetWindowUserPointer(window));
	pHandler->cursorPosCallback(xpos, ypos);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	InputHandler * pHandler = static_cast<InputHandler *>(glfwGetWindowUserPointer(window));
	pHandler->mouseButtonCallback(button, action, mods);
}
