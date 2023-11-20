#ifndef INPUT_HANDLER_H
#define INPUT_HANDLER_H


#include <GLFW/glfw3.h>


class Camera;

class InputHandler
{
public:
	InputHandler(GLFWwindow * _pWindow, Camera * _pCamera);

	void handleInputs();

	// callbacks
	void keyCallback(int key, int scancode, int action, int mods);
	void mouseButtonCallback(int button, int action, int mods);
	void cursorPosCallback(double xpos, double ypos);
private:
	GLFWwindow * pWindow;
	Camera * pCamera;

	bool cursorCaptured;
};

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void cursorPosCallback(GLFWwindow * window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

#endif // INPUT_HANDLER_H
