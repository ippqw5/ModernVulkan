#include "application.h"

static Application* s_Instance = nullptr;

Application::Application()
{
	s_Instance = this;
}

Application& Application::Get()
{
	return *s_Instance;
}