// main.cpp

#include <iostream>
#include <stdexcept>
#include <set>
#include <optional>
#include <limits>
#include <algorithm>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <GLFW/glfw3.h> // 必须在vulkan.hpp之后include

constexpr uint32_t WINDOW_WIDTH  = 800;
constexpr uint32_t WINDOW_HEIGHT = 600;

constexpr std::array<const char*, 1> REQUIRED_LAYERS{
	"VK_LAYER_KHRONOS_validation"
};

constexpr std::array<const char*, 1> DEVICE_EXTENSIONS{
	vk::KHRSwapchainExtensionName
};

#ifdef NDEBUG
constexpr bool ENABLE_VALIDATION_LAYER = false;
#else
constexpr bool ENABLE_VALIDATION_LAYER = true;
#endif

static std::vector<const char*> getRequiredExtensions() {
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
	extensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);
	if constexpr (ENABLE_VALIDATION_LAYER) {
		extensions.emplace_back(vk::EXTDebugUtilsExtensionName);
	}
	return extensions;
}

static VKAPI_ATTR uint32_t VKAPI_CALL debugMessageFunc(
	vk::DebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
	vk::DebugUtilsMessageTypeFlagsEXT              messageTypes,
	vk::DebugUtilsMessengerCallbackDataEXT const* pCallbackData,
	void* pUserData
) {
	std::println(std::cerr, "validation layer: {}", pCallbackData->pMessage);
	return false;
}

static constexpr vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo() {
	constexpr vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
	);
	constexpr vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags(
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
	);
	return { {}, severityFlags, messageTypeFlags, &debugMessageFunc };
}

static bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& physicalDevice) {
	const auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
	std::set<std::string> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}
	return requiredExtensions.empty();
}

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
			availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
			) return availableFormat;
	}
	return availableFormats.at(0);
}

static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
			return availablePresentMode;
		}
	}
	return vk::PresentModeKHR::eFifo;
}

class HelloTriangle {

public:

	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() const {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	struct SwapChainSupportDetails {
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR>  formats;
		std::vector<vk::PresentModeKHR> presentModes;
	};

	void run()
	{
		iniWindow();
		initVulkan();
		mainloop();
		cleanup();
	}

private:
	GLFWwindow* m_GLFWwindow{ nullptr };

	/*
	* 初始化 Vulkan Loader 自动加载全局函数指针
	* 必须初始化，且只能初始化一次
	* 可无参构造，且不可nullptr构造（特殊）
	*/
	vk::raii::Context m_Context; 
	/*
															   |--->  Physical Device
															   |		
	Vulkan App --->		|-------------------|  ---->  Vulkan ICD ---> Physical Device
						|					|	      ^  
						|                   |         |
	Vulkan App --->		|    Vulkan Loader	|  --------
						|					|
	Vulkan App --->		|-------------------|  ---->  Vulkan ICD ---> Physical Device

	*/

	vk::raii::Instance m_Instance{ nullptr };
	vk::raii::DebugUtilsMessengerEXT m_DebugMessenger{ nullptr };
	vk::raii::SurfaceKHR m_Surface{ nullptr };
	vk::raii::PhysicalDevice m_PhysicalDevice{ nullptr };
	vk::raii::Device m_Device{ nullptr };
	vk::raii::Queue m_GraphicsQueue{ nullptr };
	vk::raii::Queue m_PresentQueue{ nullptr };

	vk::raii::SwapchainKHR m_SwapChain{ nullptr };
	std::vector<vk::Image> m_SwapChainImages;
	vk::Format m_SwapChainImageFormat{};
	vk::Extent2D m_SwapChainExtent{};
	std::vector<vk::raii::ImageView> m_SwapChainImageViews;

	vk::raii::RenderPass m_RenderPass{ nullptr };
	std::vector<vk::raii::Framebuffer> m_SwapChainFramebuffers;

	void iniWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // 禁用默认的OpenGL backend
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // 暂时固定窗口大小
		
		m_GLFWwindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "HelloTriangle", nullptr, nullptr);
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		selectPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createFrameBuffers();
	}

	void createInstance()
	{
		if constexpr (ENABLE_VALIDATION_LAYER) {
			if (!checkValidationLayerSupport()) throw std::runtime_error("validation layer requested, but not available!");
		}

		vk::ApplicationInfo applicationInfo;
		applicationInfo
			.setPApplicationName("Hello Triangle")
			.setApplicationVersion(1)
			.setPEngineName("No Engine")
			.setEngineVersion(1)
			.setApiVersion(vk::makeApiVersion(0, 1, 3, 0));
		
		vk::InstanceCreateInfo createInfo;
		createInfo
			.setFlags({})
			.setPApplicationInfo(&applicationInfo);

		std::vector<const char*> requiredExtensions = getRequiredExtensions();
		createInfo.setPEnabledExtensionNames(requiredExtensions); //一次性设置 Extension Count 和 Extension Name
		createInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;

		constexpr auto debugMessengerCreateInfo = populateDebugMessengerCreateInfo();
		if  constexpr (ENABLE_VALIDATION_LAYER) {
			createInfo.setPEnabledLayerNames(REQUIRED_LAYERS);
			createInfo.pNext = &debugMessengerCreateInfo;
		}

		m_Instance = m_Context.createInstance(createInfo);
	}

	void setupDebugMessenger() 
	{
		if constexpr (!ENABLE_VALIDATION_LAYER) return;
		constexpr auto createInfo = populateDebugMessengerCreateInfo();
		m_DebugMessenger = m_Instance.createDebugUtilsMessengerEXT(createInfo);
	}

	void createSurface()
	{
		VkSurfaceKHR cSurface;
		if (glfwCreateWindowSurface(*m_Instance, m_GLFWwindow, nullptr, &cSurface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface");
		}
		
		m_Surface = vk::raii::SurfaceKHR(m_Instance, cSurface);
	}

	void selectPhysicalDevice()
	{
		const auto physicalDevices = m_Instance.enumeratePhysicalDevices();
		if (physicalDevices.empty()) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		for (const auto& it : physicalDevices) {
			if (isDeviceSuitable(it)) {
				m_PhysicalDevice = it;
				break;
			}
		}

		if (m_PhysicalDevice == nullptr) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}
	
	void createLogicalDevice()
	{
		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		const auto [graphics, present] = findQueueFamilies(m_PhysicalDevice);
		std::set<uint32_t> uniqueQueueFamilies = { graphics.value(), present.value() };
		
		constexpr float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			vk::DeviceQueueCreateInfo queueCreateInfo;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.setQueuePriorities(queuePriority);
			queueCreateInfos.emplace_back(queueCreateInfo);
		}

		vk::PhysicalDeviceFeatures deviceFeatures;

		vk::DeviceCreateInfo createInfo;
		createInfo.setQueueCreateInfos(queueCreateInfos)
			.setPEnabledFeatures(&deviceFeatures)
			.setPEnabledExtensionNames(DEVICE_EXTENSIONS);

		m_Device = m_PhysicalDevice.createDevice(createInfo);
		m_GraphicsQueue = m_Device.getQueue(graphics.value(), 0);
		m_PresentQueue = m_Device.getQueue(present.value(), 0);
	}

	void createSwapChain()
	{
		const auto [capabilities, formats, presentModes] = querySwapChainSupport(m_PhysicalDevice);
		const auto surfaceFormat = chooseSwapSurfaceFormat(formats);
		const auto presentMode = chooseSwapPresentMode(presentModes);
		const auto extent = chooseSwapExtent(capabilities);

		uint32_t imageCount = capabilities.minImageCount + 1;
		if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
			imageCount = capabilities.maxImageCount;
		}

		vk::SwapchainCreateInfoKHR createInfo;
		createInfo.setSurface(m_Surface)
			.setMinImageCount(imageCount)
			.setImageExtent(extent)
			.setImageFormat(surfaceFormat.format)
			.setImageColorSpace(surfaceFormat.colorSpace)
			.setPresentMode(presentMode)
			.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
			.setImageArrayLayers(1);

		const auto [graphics, present] = findQueueFamilies(m_PhysicalDevice);
		std::vector<uint32_t> queueFamilyIndices{ graphics.value(), present.value() };
		if (graphics != present) {
			createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
			createInfo.setQueueFamilyIndices(queueFamilyIndices);
		}
		else {
			createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
		}

		createInfo.setPreTransform(capabilities.currentTransform);
		createInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
		createInfo.setClipped(true);
		createInfo.setOldSwapchain(nullptr);

		m_SwapChain = m_Device.createSwapchainKHR(createInfo);
		m_SwapChainImages = m_SwapChain.getImages();
		m_SwapChainExtent = extent;
		m_SwapChainImageFormat = surfaceFormat.format;
	}

	void createImageViews()
	{
		//不能用resize，因为vk::raii::ImageView没有无参构造。用reserve仅分配空间，不实例化
		m_SwapChainImageViews.reserve(m_SwapChainImages.size());

		for (const auto& image : m_SwapChainImages)
		{
			vk::ImageViewCreateInfo createInfo;
			createInfo.setImage(image)
				.setViewType(vk::ImageViewType::e2D)
				.setFormat(m_SwapChainImageFormat);

			vk::ImageSubresourceRange range;
			range.setAspectMask(vk::ImageAspectFlagBits::eColor)
				.setBaseMipLevel(0)
				.setLevelCount(1)
				.setBaseArrayLayer(0)
				.setLayerCount(1);

			createInfo.setSubresourceRange(range);
			
			m_SwapChainImageViews.emplace_back(m_Device.createImageView(createInfo));
		}
	}

	void createRenderPass()
	{
		vk::AttachmentDescription colorAttachment;
		colorAttachment.format = m_SwapChainImageFormat;
		colorAttachment.samples = vk::SampleCountFlagBits::e1;
		colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
		colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
		colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
		colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

		vk::AttachmentReference colorAttachmentRef;
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::SubpassDescription subpass;
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;

		subpass.setColorAttachments(colorAttachmentRef);

		vk::RenderPassCreateInfo renderPassInfo;
		renderPassInfo.setAttachments(colorAttachment);
		renderPassInfo.setSubpasses(subpass);

		m_RenderPass = m_Device.createRenderPass(renderPassInfo);
	}

	void createFrameBuffers()
	{
		m_SwapChainFramebuffers.reserve(m_SwapChainImageViews.size());
		vk::FramebufferCreateInfo framebufferInfo;
		framebufferInfo.renderPass = m_RenderPass;
		framebufferInfo.width = m_SwapChainExtent.width;
		framebufferInfo.height = m_SwapChainExtent.height;
		framebufferInfo.layers = 1;
		for (const auto& imageView : m_SwapChainImageViews) {
			framebufferInfo.setAttachments(*imageView);
			m_SwapChainFramebuffers.emplace_back(m_Device.createFramebuffer(framebufferInfo));
		}
	}
private:
	bool isDeviceSuitable(const vk::raii::PhysicalDevice& physicalDevice) const
	{
		const auto indices = findQueueFamilies(physicalDevice);
		const bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);

		if (const auto indices = findQueueFamilies(physicalDevice);
			!indices.isComplete()
			) return false;

		if (const bool extensionSupport = checkDeviceExtensionSupport(physicalDevice);
			!extensionsSupported
			) return false;

		if (const auto swapChainSupportDetails = querySwapChainSupport(physicalDevice);
			swapChainSupportDetails.formats.empty() || swapChainSupportDetails.presentModes.empty()
			) return false;

		return true;
	}
	
	QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const
	{
		QueueFamilyIndices indices;
		
		// 找Graphics Queue
		const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
		for (int ii = 0; const auto& queueFamilies : queueFamilies) {
			if (queueFamilies.queueFlags & vk::QueueFlagBits::eGraphics) {
				indices.graphicsFamily = ii;
			}

			if (physicalDevice.getSurfaceSupportKHR(ii, m_Surface)) {
				indices.presentFamily = ii;
			}

			if (indices.isComplete()) break;

			++ii;
		}

		return indices;
	}

	bool checkValidationLayerSupport() const {
		const auto layers = m_Context.enumerateInstanceLayerProperties();
		std::set<std::string> requiredLayers(REQUIRED_LAYERS.begin(), REQUIRED_LAYERS.end());
		for (const auto& layer : layers) {
			requiredLayers.erase(layer.layerName);
		}
		return requiredLayers.empty();
	}

	SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& physicalDevice) const
	{
		SwapChainSupportDetails details;
		details.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(m_Surface);
		details.formats = physicalDevice.getSurfaceFormatsKHR(m_Surface);
		details.presentModes = physicalDevice.getSurfacePresentModesKHR(m_Surface);

		return details;
	}

	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(m_GLFWwindow, &width, &height);
		vk::Extent2D actualExtent(
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		);

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
		return actualExtent;
	}

private:
	void mainloop()
	{
		while (!glfwWindowShouldClose(m_GLFWwindow))
		{
			glfwPollEvents();
		}
	}
	
	void cleanup()
	{
		glfwDestroyWindow(m_GLFWwindow);
		glfwTerminate();
	}
};


int main()
{
	try
	{
		HelloTriangle app;
		app.run();
	}
	catch (const vk::SystemError& err)
	{
		std::println(std::cerr, "vk::SystemError -- code : {} ", err.code().message());
		std::println(std::cerr, "vk::SystemError -- what : {} ", err.what());
	}
	catch (const std::exception& err)
	{
		std::println(std::cerr, "Standard exception: {}", err.what());
	}
}