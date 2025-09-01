// main.cpp

#include <iostream>
#include <stdexcept>
#include <set>
#include <optional>
#include <limits>
#include <algorithm>
#include <fstream>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <GLFW/glfw3.h> // 必须在vulkan.hpp之后include

#include <glm/glm.hpp>

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	const size_t fileSize = file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close(); // optional

	return buffer;
}

constexpr uint32_t WINDOW_WIDTH  = 800;
constexpr uint32_t WINDOW_HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

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

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription() {
		vk::VertexInputBindingDescription bindingDescription;
		bindingDescription.binding = 0;
		bindingDescription.inputRate = vk::VertexInputRate::eVertex;
		bindingDescription.stride = sizeof(Vertex);
		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 2>  getAttributeDescriptions() {
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);
		attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].offset = offsetof(Vertex, color);
		attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;

		return attributeDescriptions;
	}
};

const std::vector<Vertex> vertices = {
	{{0.0f, -0.5f}, {0.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

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
	void run()
	{
		initWindow();
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
	vk::raii::PipelineLayout m_PipelineLayout{ nullptr };
	vk::raii::Pipeline m_GraphicsPipeline{ nullptr };

	vk::raii::CommandPool m_CommandPool{ nullptr };
	std::vector<vk::raii::CommandBuffer> m_CommandBuffers;

	vk::raii::DeviceMemory m_VertexBufferMemory{ nullptr };
	vk::raii::Buffer m_VertexBuffer{ nullptr };

	std::vector<vk::raii::Semaphore> m_ImageAvailableSemaphores;
	std::vector<vk::raii::Semaphore> m_RenderFinishedSemaphores;
	std::vector<vk::raii::Fence> m_InFlightFences;

	uint32_t m_CurrentFrame = 0;
	bool m_FramebufferResized = false;

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // 禁用默认的OpenGL backend
		
		m_GLFWwindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "HelloTriangle", nullptr, nullptr);
		
		glfwSetWindowUserPointer(m_GLFWwindow, this);
		glfwSetFramebufferSizeCallback(m_GLFWwindow, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		const auto app = static_cast<HelloTriangle*>(glfwGetWindowUserPointer(window));
		app->m_FramebufferResized = true;
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
		createFramebuffers();
		createGraphicsPipeline();
		createCommandPool();
		createCommandBuffers();
		createSyncObjects();
		createVertexBuffer();
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

		vk::SubpassDependency dependency;
		dependency.srcSubpass = vk::SubpassExternal;
		dependency.dstSubpass = 0;

		dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

		dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependency.srcAccessMask = {};
		renderPassInfo.setDependencies(dependency);

		m_RenderPass = m_Device.createRenderPass(renderPassInfo);
	}

	void createFramebuffers()
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

	void createGraphicsPipeline() {
		const auto vertShaderCode = readFile("shaders/graphics.vert.spv");
		const auto fragShaderCode = readFile("shaders/graphics.frag.spv");

		vk::raii::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		vk::raii::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);
	
		vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
		vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
		fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		const auto shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

		const auto dynamicStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState;
		dynamicState.setDynamicStates(dynamicStates);
		
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
		const auto bindingDescription = Vertex::getBindingDescription();
		const auto attributeDescriptions = Vertex::getAttributeDescriptions();
		vertexInputInfo.setVertexBindingDescriptions(bindingDescription);
		vertexInputInfo.setVertexAttributeDescriptions(attributeDescriptions);

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
		inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
		
		vk::PipelineViewportStateCreateInfo viewportState;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;
		
		vk::PipelineRasterizationStateCreateInfo rasterizer;
		rasterizer.polygonMode = vk::PolygonMode::eFill;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = vk::CullModeFlagBits::eBack;
		rasterizer.frontFace = vk::FrontFace::eClockwise;
		
		vk::PipelineMultisampleStateCreateInfo multisampling;
		multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
		multisampling.sampleShadingEnable = false;
		
		vk::PipelineColorBlendAttachmentState colorBlendAttachment;
		colorBlendAttachment.blendEnable = false; // default
		colorBlendAttachment.colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
		
		vk::PipelineColorBlendStateCreateInfo colorBlending;
		colorBlending.logicOpEnable = false;
		colorBlending.logicOp = vk::LogicOp::eCopy;
		colorBlending.setAttachments(colorBlendAttachment);
		
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
		m_PipelineLayout = m_Device.createPipelineLayout(pipelineLayoutInfo);
		

		vk::GraphicsPipelineCreateInfo pipelineInfo;
		pipelineInfo.setStages(shaderStages);
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; // Optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = m_PipelineLayout;
		pipelineInfo.renderPass = m_RenderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = nullptr; // Optional
		pipelineInfo.basePipelineIndex = -1; // Optional
		m_GraphicsPipeline = m_Device.createGraphicsPipeline(nullptr, pipelineInfo);
	}

	void createCommandPool()
	{
		const auto [graphicsFamily, presentFamily] = findQueueFamilies(m_PhysicalDevice);

		vk::CommandPoolCreateInfo poolInfo;
		poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		poolInfo.queueFamilyIndex = graphicsFamily.value();

		m_CommandPool = m_Device.createCommandPool(poolInfo);

	}

	void createCommandBuffers() {
		vk::CommandBufferAllocateInfo allocInfo;
		allocInfo.commandPool = m_CommandPool;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

		m_CommandBuffers = m_Device.allocateCommandBuffers(allocInfo);
	}

	void createSyncObjects() {
		constexpr vk::SemaphoreCreateInfo semaphoreInfo;
		constexpr vk::FenceCreateInfo fenceInfo(
			vk::FenceCreateFlagBits::eSignaled  // flags
		);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			m_ImageAvailableSemaphores.emplace_back(m_Device.createSemaphore(semaphoreInfo));
			m_RenderFinishedSemaphores.emplace_back(m_Device.createSemaphore(semaphoreInfo));
			m_InFlightFences.emplace_back(m_Device.createFence(fenceInfo));
		}
	}

	void createVertexBuffer()
	{
		vk::BufferCreateInfo bufferInfo;
		bufferInfo.size = sizeof(Vertex) * vertices.size();
		bufferInfo.usage = vk::BufferUsageFlags::BitsType::eVertexBuffer;
		bufferInfo.sharingMode = vk::SharingMode::eExclusive;

		m_VertexBuffer = m_Device.createBuffer(bufferInfo);

		const vk::MemoryRequirements memRequirements = m_VertexBuffer.getMemoryRequirements();

		vk::MemoryAllocateInfo allocInfo;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(
			memRequirements.memoryTypeBits,
			vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent
		);

		m_VertexBufferMemory = m_Device.allocateMemory(allocInfo);
		m_VertexBuffer.bindMemory(m_VertexBufferMemory, 0);

		void* data = m_VertexBufferMemory.mapMemory(0, bufferInfo.size);
		memcpy(data, vertices.data(), static_cast<size_t>(bufferInfo.size));
		m_VertexBufferMemory.unmapMemory();
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

	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
		vk::ShaderModuleCreateInfo createInfo;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		return m_Device.createShaderModule(createInfo);
	}

	void recordCommandBuffer(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) const {
		constexpr vk::CommandBufferBeginInfo beginInfo;
		commandBuffer.begin(beginInfo);

		vk::RenderPassBeginInfo renderPassInfo;
		renderPassInfo.renderPass = m_RenderPass;
		renderPassInfo.framebuffer = m_SwapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
		renderPassInfo.renderArea.extent = m_SwapChainExtent;
		constexpr vk::ClearValue clearColor(vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f));
		renderPassInfo.setClearValues(clearColor);

		commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_GraphicsPipeline);

		const vk::Viewport viewport(
			0.0f, 0.0f, // x, y
			static_cast<float>(m_SwapChainExtent.width),    // width
			static_cast<float>(m_SwapChainExtent.height),   // height
			0.0f, 1.0f  // minDepth maxDepth
		);
		commandBuffer.setViewport(0, viewport);

		const vk::Rect2D scissor(
			vk::Offset2D{ 0, 0 }, // offset
			m_SwapChainExtent   // extent
		);
		commandBuffer.setScissor(0, scissor);

		const std::array<vk::Buffer, 1> vertexBuffers{ m_VertexBuffer };
		constexpr std::array<vk::DeviceSize, 1> offsets{ 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

		commandBuffer.draw(static_cast<uint32_t>(vertices.size()), 1, 0, 0);

		commandBuffer.endRenderPass();
		commandBuffer.end();
	}

	void recreateSwapChain() {

		int width = 0, height = 0;
		glfwGetFramebufferSize(m_GLFWwindow, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(m_GLFWwindow, &width, &height);
			glfwWaitEvents();
		}
		m_Device.waitIdle();

		m_SwapChainFramebuffers.clear();
		m_SwapChainImageViews.clear();
		m_SwapChainImages.clear(); // optional
		m_SwapChain = nullptr;

		createSwapChain();
		createImageViews();
		createFramebuffers();

		m_FramebufferResized = false;
	}

	uint32_t findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const {
		const auto memProperties = m_PhysicalDevice.getMemoryProperties();
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
			if ((typeFilter & (1 << i)) &&
				(memProperties.memoryTypes[i].propertyFlags & properties) == properties
				) return i;
		}
		throw std::runtime_error("failed to find suitable memory type!");
		return 0; // optional
	}
private:
	void mainloop()
	{
		while (!glfwWindowShouldClose(m_GLFWwindow))
		{
			glfwPollEvents();
			drawFrame();
		}

		m_Device.waitIdle();
	}
	
	void cleanup()
	{
		glfwDestroyWindow(m_GLFWwindow);
		glfwTerminate();
	}

	void drawFrame() {
		if (const auto res = m_Device.waitForFences(*m_InFlightFences[m_CurrentFrame], true, std::numeric_limits<uint64_t>::max());
			res != vk::Result::eSuccess
			) throw std::runtime_error{ "waitForFences in drawFrame was failed" };

		uint32_t imageIndex;
		try {
			// std::pair<vk::Result, uint32_t>
			const auto [res, idx] = m_SwapChain.acquireNextImage(UINT64_MAX, m_ImageAvailableSemaphores[m_CurrentFrame]);
			imageIndex = idx;
		}
		catch (const vk::OutOfDateKHRError&) {
			recreateSwapChain();
			return;
		} // Do not catch other exceptions

		m_Device.resetFences(*m_InFlightFences[m_CurrentFrame]);

		m_CommandBuffers[m_CurrentFrame].reset();
		recordCommandBuffer(m_CommandBuffers[m_CurrentFrame], imageIndex);

		vk::SubmitInfo submitInfo;
		submitInfo.setWaitSemaphores(*m_ImageAvailableSemaphores[m_CurrentFrame]);
		std::array<vk::PipelineStageFlags, 1> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
		submitInfo.setWaitDstStageMask(waitStages);
		submitInfo.setCommandBuffers(*m_CommandBuffers[m_CurrentFrame]);

		submitInfo.setSignalSemaphores(*m_RenderFinishedSemaphores[m_CurrentFrame]);
		m_GraphicsQueue.submit(submitInfo, m_InFlightFences[m_CurrentFrame]);

		vk::PresentInfoKHR presentInfo;
		presentInfo.setWaitSemaphores(*m_RenderFinishedSemaphores[m_CurrentFrame]);
		presentInfo.setSwapchains(*m_SwapChain);
		presentInfo.pImageIndices = &imageIndex;
		try {
			const auto res = m_PresentQueue.presentKHR(presentInfo);
			if (res == vk::Result::eSuboptimalKHR) {
				recreateSwapChain();
			}
		}
		catch (const vk::OutOfDateKHRError&) {
			recreateSwapChain();
		} // Do not catch other exceptions

		if (m_FramebufferResized) {
			recreateSwapChain();
		}

		m_CurrentFrame = (m_CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
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