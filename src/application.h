#include <glfw/glfw3.h>

class Application
{
public:
	Application();
	virtual ~Application() {}

	static Application& Get();

	void SetWindowHandle(GLFWwindow* windowHandle) { m_WindowHandle = windowHandle; }
	GLFWwindow* GetWindowHandle() const { return m_WindowHandle; }

private:
	GLFWwindow* m_WindowHandle = nullptr;
};