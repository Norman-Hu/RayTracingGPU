#include <InputHandler.cuh>
#include <Camera.h>
#include <Vec3.cuh>


InputHandler::InputHandler(GLFWwindow * _pWindow, Camera * _pCamera)
: pWindow(_pWindow)
, pCamera(_pCamera)
, cursorCaptured(false)
, lastCurPosX(0.0f)
, lastCurPosY(0.0f)
{
	if (glfwRawMouseMotionSupported())
		glfwSetInputMode(pWindow, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
}

void InputHandler::handleInputs()
{
	constexpr float speed = 0.05f;

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

    float up_down = 0.0f;
    if (glfwGetKey(pWindow, GLFW_KEY_Q))
        up_down -= 1.f;
    if (glfwGetKey(pWindow, GLFW_KEY_E))
        up_down += 1.f;

    if (vertical != 0.0f || horizontal != 0.0f || up_down != 0.0f)
    {
        Vec3 movement = pCamera->Front*vertical + pCamera->Right*horizontal + pCamera->WorldUp*up_down;
        movement.normalize();
        movement *= speed;
        pCamera->Position += movement;
    }
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

		double x, y;
		glfwGetCursorPos(pWindow, &x, &y);
		lastCurPosX = x;
		lastCurPosY = y;
	}
}

void InputHandler::cursorPosCallback(double xpos, double ypos)
{
	if (!cursorCaptured)
		return;

	float offsetX = xpos - lastCurPosX;
	float offsetY = ypos - lastCurPosY;
	lastCurPosX = xpos;
	lastCurPosY = ypos;
	pCamera->ProcessMouseMovement(offsetX, -offsetY, true);
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
