// main.cpp

#include <iostream>
#include <stdexcept>
#include <set>
#include <optional>
#include <limits>
#include <algorithm>
#include <fstream>
#include <print>
#include <chrono>
#include <map>
#include <tuple>
#include <random>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <GLFW/glfw3.h> // 必须在vulkan.hpp之后include

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Vulkan 的深度范围是[0, 1]，OpenGL 的是[-1, 1]
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "application.h"
#include "camera.h"

const std::string BUNNY_PATH = "models/bunny.obj";
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
const std::string CRATE_MODEL_PATH = "models/crate.obj";
const std::string CRATE_TEXTURE_PATH = "textures/crate.jpg";

const uint32_t BUNNY_NUMBER = 5;

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
	std::optional<uint32_t> transferFamily;

	bool isComplete() const {
		return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR>  formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription() {
		vk::VertexInputBindingDescription bindingDescription;
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = vk::VertexInputRate::eVertex;

		return bindingDescription;
	}
	static std::array<vk::VertexInputAttributeDescription, 3>  getAttributeDescriptions() {
		std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions;

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

	bool operator<(const Vertex& other) const {
		return std::tie(pos.x, pos.y, pos.z, color.x, color.y, color.z, texCoord.x, texCoord.y)
			< std::tie(other.pos.x, other.pos.y, other.pos.z, other.color.x, other.color.y, other.color.z, other.texCoord.x, other.texCoord.y);
	}
};

struct UniformBufferObject {
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct InstanceData {
	glm::mat4 model;

	static vk::VertexInputBindingDescription getBindingDescription() {
		vk::VertexInputBindingDescription bindingDescription;
		bindingDescription.binding = 1;
		bindingDescription.stride = sizeof(InstanceData);
		bindingDescription.inputRate = vk::VertexInputRate::eInstance;

		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 4>  getAttributeDescriptions() {
		std::array<vk::VertexInputAttributeDescription, 4> attributeDescriptions;
		for (uint32_t i = 0; i < 4; ++i) {
			attributeDescriptions[i].binding = 1; // binding 1 for instance data
			attributeDescriptions[i].location = 3 + i; // location 3, 4, 5, 6
			attributeDescriptions[i].format = vk::Format::eR32G32B32A32Sfloat;
			attributeDescriptions[i].offset = sizeof(glm::vec4) * i;
		}
		return attributeDescriptions;
	}
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

class HelloTriangle : Application {

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
	初始化 Vulkan Loader 自动加载全局函数指针
	必须初始化，且只能初始化一次
	可无参构造，且不可nullptr构造（特殊）
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
	vk::raii::Queue m_TransferQueue{ nullptr };

	vk::raii::SwapchainKHR m_SwapChain{ nullptr };
	std::vector<vk::Image> m_SwapChainImages;
	vk::Format m_SwapChainImageFormat{};
	vk::Extent2D m_SwapChainExtent{};
	std::vector<vk::raii::ImageView> m_SwapChainImageViews;

	vk::raii::DeviceMemory m_DepthImageMemory{ nullptr };
	vk::raii::Image m_DepthImage{ nullptr };
	vk::raii::ImageView m_DepthImageView{ nullptr };

	vk::raii::DeviceMemory m_ColorImageMemory{ nullptr };
	vk::raii::Image m_ColorImage{ nullptr };
	vk::raii::ImageView m_ColorImageView{ nullptr };

	vk::raii::RenderPass m_RenderPass{ nullptr };
	std::vector<vk::raii::Framebuffer> m_SwapChainFramebuffers;
	std::vector<vk::raii::DescriptorSetLayout> m_DescriptorSetLayouts;
	vk::raii::PipelineLayout m_PipelineLayout{ nullptr };
	vk::raii::Pipeline m_GraphicsPipeline{ nullptr };

	vk::raii::CommandPool m_CommandPool{ nullptr };
	vk::raii::CommandPool m_TransferCommandPool{ nullptr };
	std::vector<vk::raii::CommandBuffer> m_CommandBuffers;

	vk::raii::DeviceMemory m_VertexBufferMemory{ nullptr };
	vk::raii::Buffer m_VertexBuffer{ nullptr };
	vk::raii::DeviceMemory m_InstanceBufferMemory{ nullptr };
	vk::raii::Buffer m_InstanceBuffer{ nullptr };

	std::vector<Vertex> m_MeshVertices;
	std::vector<uint32_t> m_MeshIndices;
	std::vector<uint32_t> m_MeshIndexOffsets;
	std::vector<uint32_t> m_MeshIndexCounts;
	std::vector<InstanceData> m_InstanceDatas;
	std::vector<glm::mat4> m_DynamicUboMatrices;

	vk::raii::DeviceMemory m_IndexBufferMemory{ nullptr };
	vk::raii::Buffer m_IndexBuffer{ nullptr };
	
	std::vector<vk::raii::DeviceMemory> m_UniformBuffersMemory;
	std::vector<vk::raii::Buffer> m_UniformBuffers;
	std::vector<void*> m_UniformBuffersMapped;

	std::vector<vk::raii::DeviceMemory> m_DynamicUniformBuffersMemory;
	std::vector<vk::raii::Buffer> m_DynamicUniformBuffers;
	std::vector<void*> m_DynamicUniformBuffersMapped;

	uint32_t m_MipLevels = 1;
	std::vector<vk::raii::DeviceMemory> m_TextureImageMemorys;
	std::vector<vk::raii::Image> m_TextureImages;
	std::vector<vk::raii::ImageView> m_TextureImageViews;
	vk::raii::Sampler m_TextureSampler{ nullptr };

	vk::raii::DescriptorPool m_DescriptorPool{ nullptr };
	std::vector<vk::raii::DescriptorSet> m_DescriptorSets;
	vk::raii::DescriptorSet m_CombinedDescriptorSet{ nullptr };

	std::vector<vk::raii::Semaphore> m_ImageAvailableSemaphores;
	std::vector<vk::raii::Semaphore> m_RenderFinishedSemaphores;
	std::vector<vk::raii::Fence> m_InFlightFences;

	vk::SampleCountFlagBits m_MsaaSamples = vk::SampleCountFlagBits::e1;

	uint32_t m_CurrentFrame = 0;
	bool m_FramebufferResized = false;

	Camera m_Camera{ 45.0f, 0.1f, 100.0f };

private:
	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // 禁用 OpenGL Backend (默认是启用的)
		
		m_GLFWwindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "HelloTriangle", nullptr, nullptr);
		
		m_Camera.OnResize(WINDOW_WIDTH, WINDOW_HEIGHT);
		SetWindowHandle(m_GLFWwindow);

		glfwSetWindowUserPointer(m_GLFWwindow, this);
		glfwSetFramebufferSizeCallback(m_GLFWwindow, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		const auto app = static_cast<HelloTriangle*>(glfwGetWindowUserPointer(window));
		app->m_FramebufferResized = true;
	}

	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
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
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createCommandBuffers();
		//createTransferCommandPool();
		createDepthResources();
		createColorResources();
		createFramebuffers();
		createTextureImage(TEXTURE_PATH);
		createTextureImage(CRATE_TEXTURE_PATH);
		createTextureImageView();
		createTextureSampler();
		createSyncObjects();
		loadModel(MODEL_PATH);
		loadModel(BUNNY_PATH);
		loadModel(CRATE_MODEL_PATH);
		initInstanceDatas();
		initDynamicUboMatrices();
		createVertexBuffer();
		createInstanceBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDynamicUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
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
				m_MsaaSamples = getMaxUsableSampleCount();
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
		const auto [graphics, present, transfer] = findQueueFamilies(m_PhysicalDevice);
		std::set<uint32_t> uniqueQueueFamilies = { graphics.value(), present.value(), transfer.value() };
		
		constexpr float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			vk::DeviceQueueCreateInfo queueCreateInfo;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.setQueuePriorities(queuePriority);
			queueCreateInfos.emplace_back(queueCreateInfo);
		}

		vk::PhysicalDeviceFeatures deviceFeatures;
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.sampleRateShading = VK_TRUE;

		vk::DeviceCreateInfo createInfo;
		createInfo.setQueueCreateInfos(queueCreateInfos)
			.setPEnabledFeatures(&deviceFeatures)
			.setPEnabledExtensionNames(DEVICE_EXTENSIONS);

		m_Device = m_PhysicalDevice.createDevice(createInfo);
		m_GraphicsQueue = m_Device.getQueue(graphics.value(), 0);
		m_PresentQueue = m_Device.getQueue(present.value(), 0);
		m_TransferQueue = m_Device.getQueue(transfer.value(), 0);
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

		const auto [graphics, present, transfer] = findQueueFamilies(m_PhysicalDevice);
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

	void createImageViews() {
		m_SwapChainImageViews.reserve(m_SwapChainImages.size());
		for (const auto& image : m_SwapChainImages) {
			m_SwapChainImageViews.emplace_back(
				createImageView(
					image,
					m_SwapChainImageFormat,
					vk::ImageAspectFlagBits::eColor,
					1)
			);
		}
	}

	void createRenderPass()
	{
		// RenderPass_CreateInfo
		vk::RenderPassCreateInfo renderPassInfo;

		// Attachments
		vk::AttachmentDescription colorAttachment;
		colorAttachment.format = m_SwapChainImageFormat;
		colorAttachment.samples = m_MsaaSamples;
		colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
		colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
		colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
		colorAttachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::AttachmentDescription colorAttachmentResolve;
		colorAttachmentResolve.format = m_SwapChainImageFormat;
		colorAttachmentResolve.samples = vk::SampleCountFlagBits::e1;
		colorAttachmentResolve.loadOp = vk::AttachmentLoadOp::eDontCare;
		colorAttachmentResolve.storeOp = vk::AttachmentStoreOp::eStore;
		colorAttachmentResolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		colorAttachmentResolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		colorAttachmentResolve.initialLayout = vk::ImageLayout::eUndefined;
		colorAttachmentResolve.finalLayout = vk::ImageLayout::ePresentSrcKHR;

		vk::AttachmentDescription depthAttachment;
		depthAttachment.format = findDepthFormat({ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint });
		depthAttachment.samples = m_MsaaSamples;
		depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
		depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
		depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
		depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		const auto attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
		renderPassInfo.setAttachments(attachments);
		// Attachments

		// SubPass
		vk::AttachmentReference colorAttachmentRef;
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
		vk::AttachmentReference depthAttachmentRef;
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		vk::AttachmentReference colorAttachmentResolveRef;
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::SubpassDescription subpass;
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pResolveAttachments = &colorAttachmentResolveRef;
		subpass.colorAttachmentCount = 1; // ColorAttachment 和 ResolveAttachment 共享同一个count

		renderPassInfo.setSubpasses(subpass);
		// SubPass

		// SubPass Dependency
		vk::SubpassDependency dependency;
		dependency.srcSubpass = vk::SubpassExternal;
		dependency.dstSubpass = 0;

		dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
		dependency.srcAccessMask = {};
		dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
		dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
		renderPassInfo.setDependencies(dependency);
		// SubPass Dependency

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
		for (const auto& swapchainImageView : m_SwapChainImageViews) {
			const std::array<vk::ImageView, 3> imageViews{ 
				m_ColorImageView,
				m_DepthImageView,
				swapchainImageView,
			};
			framebufferInfo.setAttachments(imageViews);
			m_SwapChainFramebuffers.emplace_back(m_Device.createFramebuffer(framebufferInfo));
		}
	}

	void createDescriptorSetLayout()
	{
		// 多个描述符集绑定不同类型的资源，并在着色器用 set = 区分描述符集。 
		// 一个描述符集绑定多个资源，通过 binding 区分。

		// set 0: UBO + Dynamic UBO
		vk::DescriptorSetLayoutBinding uboLayoutBinding;
		uboLayoutBinding.binding = 0; // 对应着色器中 layout(binding = 0)
		uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

		vk::DescriptorSetLayoutBinding dynamicUboLayoutBinding;
		dynamicUboLayoutBinding.binding = 1; // 对应着色器中 layout(binding = 1)
		dynamicUboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
		dynamicUboLayoutBinding.descriptorCount = 1;
		dynamicUboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

		const auto uboLayoutBindings = { uboLayoutBinding, dynamicUboLayoutBinding };
		vk::DescriptorSetLayoutCreateInfo layoutInfo;
		layoutInfo.setBindings(uboLayoutBindings);
		m_DescriptorSetLayouts.emplace_back(m_Device.createDescriptorSetLayout(layoutInfo));

		// set 1: 纹理采样器 + 纹理图像视图
		vk::DescriptorSetLayoutBinding samplerLayoutBinding;
		samplerLayoutBinding.binding = 0; 
		samplerLayoutBinding.descriptorType = vk::DescriptorType::eSampler;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

		vk::DescriptorSetLayoutBinding imageLayoutBinding;
		imageLayoutBinding.binding = 1;
		imageLayoutBinding.descriptorType = vk::DescriptorType::eSampledImage;
		imageLayoutBinding.descriptorCount = 2;
		imageLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

		const auto samplerLayoutBindings = { samplerLayoutBinding, imageLayoutBinding };
		vk::DescriptorSetLayoutCreateInfo samplerLayoutInfo;
		samplerLayoutInfo.setBindings(samplerLayoutBindings);
		m_DescriptorSetLayouts.emplace_back(m_Device.createDescriptorSetLayout(samplerLayoutInfo));
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
		const auto vertexBindingDescription = Vertex::getBindingDescription();
		const auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();
		const auto instanceBindingDescription = InstanceData::getBindingDescription();
		const auto instanceAttributeDescriptions = InstanceData::getAttributeDescriptions();

		std::vector<vk::VertexInputBindingDescription> bindingDescriptions = {
			vertexBindingDescription,
			instanceBindingDescription
		};

		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions(
			vertexAttributeDescriptions.begin(),
			vertexAttributeDescriptions.end()
		);
		attributeDescriptions.insert(
			attributeDescriptions.end(),
			instanceAttributeDescriptions.begin(),
			instanceAttributeDescriptions.end()
		);

		vertexInputInfo.setVertexBindingDescriptions(bindingDescriptions);
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
		rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
		
		vk::PipelineMultisampleStateCreateInfo multisampling;
		multisampling.rasterizationSamples = m_MsaaSamples;
		multisampling.sampleShadingEnable = true;
		multisampling.minSampleShading = .2f;
		
		vk::PipelineColorBlendAttachmentState colorBlendAttachment;
		colorBlendAttachment.blendEnable = false; // default
		colorBlendAttachment.colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
		
		vk::PipelineColorBlendStateCreateInfo colorBlending;
		colorBlending.logicOpEnable = false;
		colorBlending.logicOp = vk::LogicOp::eCopy;
		colorBlending.setAttachments(colorBlendAttachment);
		
		vk::PushConstantRange pushConstantRange;
		pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eFragment;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int32_t);

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
		const std::vector<vk::DescriptorSetLayout> descriptorSetLayouts(m_DescriptorSetLayouts.begin(), m_DescriptorSetLayouts.end());
		pipelineLayoutInfo.setSetLayouts(descriptorSetLayouts);
		pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);
		m_PipelineLayout = m_Device.createPipelineLayout(pipelineLayoutInfo);
		
		vk::PipelineDepthStencilStateCreateInfo depthStencil;
		depthStencil.depthTestEnable = true;
		depthStencil.depthWriteEnable = true;
		depthStencil.depthCompareOp = vk::CompareOp::eLess;
		depthStencil.depthBoundsTestEnable = false; // Optional
		depthStencil.minDepthBounds = 0.0f; // Optional if depthBoundsTestEnable is false
		depthStencil.maxDepthBounds = 1.0f; // Optional if depthBoundsTestEnable is false
		depthStencil.stencilTestEnable = false; // Optional
		depthStencil.front = vk::StencilOpState{}; // Optional if stencilTestEnable is false
		depthStencil.back = vk::StencilOpState{}; // Optional if stencilTestEnable is false

		vk::GraphicsPipelineCreateInfo pipelineInfo;
		pipelineInfo.setStages(shaderStages);
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
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
		const auto [graphicsFamily, presentFamily, transferFamily] = findQueueFamilies(m_PhysicalDevice);

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

	void createTransferCommandPool()
	{
		const auto [graphics, present, transfer] = findQueueFamilies(m_PhysicalDevice);

		vk::CommandPoolCreateInfo poolInfo;
		poolInfo.flags = vk::CommandPoolCreateFlagBits::eTransient;
		poolInfo.queueFamilyIndex = transfer.value();

		m_TransferCommandPool = m_Device.createCommandPool(poolInfo);
	}

	void createColorResources() {
		createImage(
			m_SwapChainExtent.width,
			m_SwapChainExtent.height,
			1,
			m_MsaaSamples,
			m_SwapChainImageFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransientAttachment |
			vk::ImageUsageFlagBits::eColorAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			m_ColorImage,
			m_ColorImageMemory
		);
		m_ColorImageView = createImageView(
			m_ColorImage,
			m_SwapChainImageFormat,
			vk::ImageAspectFlagBits::eColor,
			1
		);
	}

	void createDepthResources()
	{
		const vk::Format depthFormat = findDepthFormat(
			{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint }
		);

		createImage(
			m_SwapChainExtent.width,
			m_SwapChainExtent.height,
			1,
			m_MsaaSamples,
			depthFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			m_DepthImage,
			m_DepthImageMemory
		);
		m_DepthImageView = createImageView(m_DepthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
	}

	void createTextureImage(const std::string& texture_path)
	{
		int texWidth, texHeight, texChannels;
		// STBI_rgb_alpha 让他强制加载4通道，缺少的通道会自动补齐。
		stbi_uc* pixels = stbi_load(texture_path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		if (!pixels) throw std::runtime_error("failed to load texture image!");

		m_MipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

		const vk::DeviceSize imageSize = texWidth * texHeight * 4;

		vk::raii::DeviceMemory stagingBufferMemory{ nullptr };
		vk::raii::Buffer stagingBuffer{ nullptr };

		createBuffer(
			imageSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, pixels, imageSize);
		stagingBufferMemory.unmapMemory();
		stbi_image_free(pixels);

		// 临时存放纹理图像的缓冲区，创建后移动到成员变量的数组中
		vk::raii::Image tmpTextureBuffer{ nullptr };
		vk::raii::DeviceMemory tmpTextureBufferMemory{ nullptr };

		createImage(
			texWidth,
			texHeight,
			m_MipLevels,
			vk::SampleCountFlagBits::e1,
			vk::Format::eR8G8B8A8Srgb,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferSrc |
			vk::ImageUsageFlagBits::eTransferDst |
			vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			tmpTextureBuffer,
			tmpTextureBufferMemory
		);

		transitionImageLayout(
			tmpTextureBuffer,
			vk::Format::eR8G8B8A8Srgb,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			m_MipLevels
		);

		copyBufferToImage(
			stagingBuffer,
			tmpTextureBuffer,
			static_cast<uint32_t>(texWidth),
			static_cast<uint32_t>(texHeight)
		);

		generateMipmaps(
			tmpTextureBuffer,
			vk::Format::eR8G8B8A8Srgb,
			texWidth,
			texHeight,
			m_MipLevels
		);

		m_TextureImages.emplace_back(std::move(tmpTextureBuffer));
		m_TextureImageMemorys.emplace_back(std::move(tmpTextureBufferMemory));
	}

	void createTextureImageView()
	{
		for (const auto& image : m_TextureImages)
		{
			m_TextureImageViews.emplace_back(
				createImageView(
					*image,
					vk::Format::eR8G8B8A8Srgb,
					vk::ImageAspectFlagBits::eColor,
					m_MipLevels
				)
			);
		}
	}

	void createTextureSampler()
	{
		vk::SamplerCreateInfo samplerInfo;
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;

		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

		samplerInfo.anisotropyEnable = true;
		samplerInfo.maxAnisotropy = m_PhysicalDevice.getProperties().limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
		samplerInfo.unnormalizedCoordinates = false;

		samplerInfo.compareEnable = false;
		samplerInfo.compareOp = vk::CompareOp::eAlways;

		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = static_cast<float>(m_MipLevels);

		m_TextureSampler = m_Device.createSampler(samplerInfo);
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

	void loadModel(const std::string& model_path)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, model_path.c_str())) {
			throw std::runtime_error(warn + err);
		}
		
		m_MeshIndexOffsets.push_back(m_MeshIndices.size()); // 记录开始位置

		std::map<Vertex, uint32_t> uniqueVertices{};

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				Vertex vertex{};

				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				// 检查是否有纹理坐标
				if (!attrib.texcoords.empty() && index.texcoord_index >= 0) {
					vertex.texCoord = {
						attrib.texcoords[2 * index.texcoord_index],
						1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
					};
				}
				else {
					vertex.texCoord = { 0.0f, 0.0f };
				}

				vertex.color = { 1.0f, 1.0f, 1.0f };

				if (!uniqueVertices.contains(vertex)) {
					uniqueVertices[vertex] = static_cast<uint32_t>(m_MeshVertices.size());
					m_MeshVertices.push_back(vertex);
				}

				m_MeshIndices.push_back(uniqueVertices[vertex]);
			}
		}

		m_MeshIndexCounts.push_back(m_MeshIndices.size() - m_MeshIndexOffsets.back()); // 索引总数减去开始位置得到mesh的索引数
	}

	void initInstanceDatas()
	{
		InstanceData instanceData{};
		m_InstanceDatas.reserve(BUNNY_NUMBER + 1);
		// 房间的旋转矩阵，参考 `updateUniformBuffer` 中的内容。
		instanceData.model = glm::rotate(
			glm::mat4(1.0f),
			glm::radians(-90.0f),
			glm::vec3(1.0f, 0.0f, 0.0f)
		) * glm::rotate(
			glm::mat4(1.0f),
			glm::radians(-90.0f),
			glm::vec3(0.0f, 0.0f, 1.0f)
		);
		m_InstanceDatas.emplace_back(instanceData);
		// 随机数生成器
		std::random_device rd;
		std::default_random_engine gen(rd());
		std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
		// 初始化兔子的参数
		for (int i = 0; i < BUNNY_NUMBER; ++i) {
			// 随机移动和水平旋转
			instanceData.model = glm::translate(
				glm::mat4(1.0f),
				glm::vec3(dis(gen), dis(gen), dis(gen))
			) * glm::rotate(
				glm::mat4(1.0f),
				glm::radians(dis(gen) * 180.0f),
				glm::vec3(0.0f, 1.0f, 0.0f)
			);
			m_InstanceDatas.emplace_back(instanceData);
		}
		instanceData.model = glm::translate(
			glm::mat4(1.0f),
			glm::vec3(0.0f, 0.0f, 1.2f)
		) * glm::scale(
			glm::mat4(1.0f),
			glm::vec3(0.2f, 0.2f, 0.2f)
		);
		m_InstanceDatas.emplace_back(instanceData);
	}

	void initDynamicUboMatrices()
	{
		m_DynamicUboMatrices.emplace_back(1.0f);
		m_DynamicUboMatrices.emplace_back(1.0f);
		m_DynamicUboMatrices.emplace_back(1.0f);
	}

	void createVertexBuffer()
	{
		const vk::DeviceSize bufferSize = sizeof(Vertex) * m_MeshVertices.size();


		vk::raii::DeviceMemory stagingBufferMemory{ nullptr };
		vk::raii::Buffer stagingBuffer{ nullptr };
		createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, m_MeshVertices.data(), static_cast<size_t>(bufferSize));
		stagingBufferMemory.unmapMemory();

		createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			m_VertexBuffer,
			m_VertexBufferMemory
		);

		copyBuffer(stagingBuffer, m_VertexBuffer, bufferSize);
	}
	
	void createInstanceBuffer()
	{
		const vk::DeviceSize bufferSize = sizeof(InstanceData) * m_InstanceDatas.size();

		vk::raii::DeviceMemory stagingBufferMemory{ nullptr };
		vk::raii::Buffer stagingBuffer{ nullptr };
		createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible |
			vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, m_InstanceDatas.data(), static_cast<size_t>(bufferSize));
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize,
			vk::BufferUsageFlagBits::eTransferDst |
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			m_InstanceBuffer,
			m_InstanceBufferMemory
		);

		copyBuffer(stagingBuffer, m_InstanceBuffer, bufferSize);
	}
	
	void createIndexBuffer()
	{
		const vk::DeviceSize bufferSize = sizeof(uint32_t) * m_MeshIndices.size();

		vk::raii::DeviceMemory stagingBufferMemory{ nullptr };
		vk::raii::Buffer stagingBuffer{ nullptr };
		createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, m_MeshIndices.data(), static_cast<size_t>(bufferSize));
		stagingBufferMemory.unmapMemory();

		createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			m_IndexBuffer,
			m_IndexBufferMemory
		);

		copyBuffer(stagingBuffer, m_IndexBuffer, bufferSize);
	}

	void createUniformBuffers()
	{
		constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

		m_UniformBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
		m_UniformBuffersMemory.reserve(MAX_FRAMES_IN_FLIGHT);
		m_UniformBuffersMapped.reserve(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			m_UniformBuffers.emplace_back(nullptr);
			m_UniformBuffersMemory.emplace_back(nullptr);
			m_UniformBuffersMapped.emplace_back(nullptr);

			createBuffer(bufferSize,
				vk::BufferUsageFlagBits::eUniformBuffer,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				m_UniformBuffers[i],
				m_UniformBuffersMemory[i]
			);

			m_UniformBuffersMapped[i] = m_UniformBuffersMemory[i].mapMemory(0, bufferSize);
		}
	}

	void createDynamicUniformBuffers()
	{
		const vk::DeviceSize bufferSize = sizeof(glm::mat4) * m_DynamicUboMatrices.size();

		m_DynamicUniformBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
		m_DynamicUniformBuffersMemory.reserve(MAX_FRAMES_IN_FLIGHT);
		m_DynamicUniformBuffersMapped.reserve(MAX_FRAMES_IN_FLIGHT);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			m_DynamicUniformBuffers.emplace_back(nullptr);
			m_DynamicUniformBuffersMemory.emplace_back(nullptr);
			m_DynamicUniformBuffersMapped.emplace_back(nullptr);
			createBuffer(bufferSize,
				vk::BufferUsageFlagBits::eUniformBuffer,
				vk::MemoryPropertyFlagBits::eHostVisible |
				vk::MemoryPropertyFlagBits::eHostCoherent,
				m_DynamicUniformBuffers[i],
				m_DynamicUniformBuffersMemory[i]
			);

			m_DynamicUniformBuffersMapped[i] = m_DynamicUniformBuffersMemory[i].mapMemory(0, bufferSize);
		}
	}

	void createDescriptorPool()
	{
		std::array<vk::DescriptorPoolSize, 4> poolSizes;
		poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = vk::DescriptorType::eSampler;
		poolSizes[1].descriptorCount = 1;
		poolSizes[2].type = vk::DescriptorType::eUniformBufferDynamic;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[3].type = vk::DescriptorType::eSampledImage;
		poolSizes[3].descriptorCount = 2;

		vk::DescriptorPoolCreateInfo poolInfo;
		poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) + 1;

		m_DescriptorPool = m_Device.createDescriptorPool(poolInfo);
	}

	void createDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_DescriptorSetLayouts[0]);
		vk::DescriptorSetAllocateInfo allocInfo;
		allocInfo.descriptorPool = m_DescriptorPool;
		allocInfo.setSetLayouts(layouts);

		m_DescriptorSets = m_Device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo bufferInfo;
			bufferInfo.buffer = m_UniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			vk::DescriptorBufferInfo dynamicBufferInfo;
			dynamicBufferInfo.buffer = m_DynamicUniformBuffers[i];
			dynamicBufferInfo.offset = 0;
			dynamicBufferInfo.range = sizeof(glm::mat4);

			std::array<vk::WriteDescriptorSet, 2> descriptorWrites;
			descriptorWrites[0].dstSet = m_DescriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
			descriptorWrites[0].setBufferInfo(bufferInfo);
			descriptorWrites[1].dstSet = m_DescriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = vk::DescriptorType::eUniformBufferDynamic;
			descriptorWrites[1].setBufferInfo(dynamicBufferInfo);

			m_Device.updateDescriptorSets(descriptorWrites, nullptr);
		}

		allocInfo.setSetLayouts(*m_DescriptorSetLayouts[1]); // 需要一次 * 显式转换
		std::vector<vk::raii::DescriptorSet> sets = m_Device.allocateDescriptorSets(allocInfo);
		m_CombinedDescriptorSet = std::move(sets.at(0));

		vk::DescriptorImageInfo samplerInfo;
		samplerInfo.sampler = m_TextureSampler;

		std::array<vk::DescriptorImageInfo, 2> textureInfos;
		for (size_t index = 0; auto& info : textureInfos) {
			info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			info.imageView = m_TextureImageViews[index];
			++index;
		}

		std::array<vk::WriteDescriptorSet, 2> combinedDescriptorWrites;
		combinedDescriptorWrites[0].dstSet = m_CombinedDescriptorSet;
		combinedDescriptorWrites[0].dstBinding = 0;
		combinedDescriptorWrites[0].dstArrayElement = 0;
		combinedDescriptorWrites[0].descriptorType = vk::DescriptorType::eSampler;
		combinedDescriptorWrites[0].setImageInfo(samplerInfo);
		combinedDescriptorWrites[1].dstSet = m_CombinedDescriptorSet;
		combinedDescriptorWrites[1].dstBinding = 1;
		combinedDescriptorWrites[1].dstArrayElement = 0;
		combinedDescriptorWrites[1].descriptorType = vk::DescriptorType::eSampledImage;
		combinedDescriptorWrites[1].setImageInfo(textureInfos);

		m_Device.updateDescriptorSets(combinedDescriptorWrites, nullptr);
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

		if (const auto supportedFeatures = physicalDevice.getFeatures();
			!supportedFeatures.samplerAnisotropy
			) return false;

		return true;
	}
	
	QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const
	{
		QueueFamilyIndices indices;

		const auto queueFamilies = physicalDevice.getQueueFamilyProperties();

		// First pass: find dedicated transfer queue
		for (uint32_t i = 0; i < queueFamilies.size(); i++)
		{
			const auto& queueFamily = queueFamilies[i];
			if ((queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) &&
				!(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics))
			{
				indices.transferFamily = i;
				break;
			}
		}

		// Second pass: find graphics and present queues
		for (uint32_t i = 0; i < queueFamilies.size(); i++) {
			const auto& queueFamily = queueFamilies[i];

			if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
			{
				indices.graphicsFamily = i;

				// Use graphics queue as transfer if no dedicated transfer queue found
				if (!indices.transferFamily.has_value())
					indices.transferFamily = i;
			}

			if (physicalDevice.getSurfaceSupportKHR(i, m_Surface)) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) break;
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
		std::array<vk::ClearValue, 2> clearValues;
		clearValues[0].color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };
		renderPassInfo.setClearValues(clearValues);

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

		const std::array<vk::Buffer, 2> vertexBuffers{ m_VertexBuffer, m_InstanceBuffer };
		// m_VertexBuffer 在数组的 0 号位，m_InstanceBuffer 在 1 号位，这就是为什么之前的 binding 参数分别是 0 和 1 。
		constexpr std::array<vk::DeviceSize, 2> offsets{ 0, 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
		commandBuffer.bindIndexBuffer(m_IndexBuffer, 0, vk::IndexType::eUint32);

		const std::array<vk::DescriptorSet, 2> descriptorSets{
			m_DescriptorSets[m_CurrentFrame],
			m_CombinedDescriptorSet
		};

		uint32_t dynamicOffset = 0; // 通过动态偏移量来选择动态 UBO 中的矩阵
		uint32_t enableTexture = 0;
		commandBuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			m_PipelineLayout,
			0,
			descriptorSets,
			dynamicOffset
		);
		commandBuffer.pushConstants<int32_t>(
			m_PipelineLayout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			enableTexture
		);
		commandBuffer.drawIndexed(  // 绘制房屋模型
			m_MeshIndexCounts[0],        // vertexCount 一个实例包含的顶点/索引数量
			1,                      // instanceCount 实例数量
			m_MeshIndexOffsets[0],      // firstIndex   索引的开始位置
			0,                      // vertexOffset 顶点的偏移量
			0                       // firstInstance 实例的开始位置
		);

		dynamicOffset = sizeof(glm::mat4);
		enableTexture = -1; //绘制兔子，无纹理，索引用 - 1
		commandBuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			m_PipelineLayout,
			0,
			descriptorSets,
			dynamicOffset
		);
		commandBuffer.pushConstants<int32_t>(
			m_PipelineLayout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			enableTexture
		);
		commandBuffer.drawIndexed(  // 绘制 BUNNY_NUMBER 个兔子模型
			m_MeshIndexCounts[1],
			BUNNY_NUMBER,
			m_MeshIndexOffsets[1],
			0,
			1
		);

		dynamicOffset = 2 * sizeof(glm::mat4);
		enableTexture = 1;  // 绘制正方体，纹理索引是 1
		commandBuffer.bindDescriptorSets( // 保持模型静止
			vk::PipelineBindPoint::eGraphics,
			m_PipelineLayout,
			0,
			descriptorSets,
			dynamicOffset
		);
		commandBuffer.pushConstants<int32_t>(
			m_PipelineLayout,
			vk::ShaderStageFlagBits::eFragment,
			0,              // offset
			enableTexture   // value
		);
		commandBuffer.drawIndexed( // draw the crate
			m_MeshIndexCounts[2],
			1,
			m_MeshIndexOffsets[2],
			0,
			BUNNY_NUMBER + 1  // 实例索引
		);

		commandBuffer.endRenderPass();
		commandBuffer.end();
	}

	void recreateSwapChain() {

		int width = 0, height = 0;
		glfwGetFramebufferSize(m_GLFWwindow, &width, &height);
		m_Camera.OnResize(width, height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(m_GLFWwindow, &width, &height);
			glfwWaitEvents();
		}
		m_Device.waitIdle();

		m_SwapChainFramebuffers.clear();
		m_SwapChainImageViews.clear();
		m_SwapChainImages.clear(); // optional
		m_SwapChain = nullptr;

		m_ColorImageView = nullptr;
		m_ColorImage = nullptr;
		m_ColorImageMemory = nullptr;

		m_DepthImageView = nullptr;
		m_DepthImage = nullptr;
		m_DepthImageMemory = nullptr;

		createSwapChain();
		createImageViews();
		createDepthResources();
		createColorResources();
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

	void createBuffer(
		const vk::DeviceSize size,
		const vk::BufferUsageFlags usage,
		const vk::MemoryPropertyFlags properties,
		vk::raii::Buffer& buffer,
		vk::raii::DeviceMemory& bufferMemory
	)
	{
		const auto [graphics, present, transfer] = findQueueFamilies(m_PhysicalDevice);

		vk::BufferCreateInfo bufferInfo;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		
		if (graphics.value() != transfer.value()) {
			std::array<uint32_t, 2> queueFamilyIndices = { graphics.value(), transfer.value() };
			bufferInfo.sharingMode = vk::SharingMode::eConcurrent;
			bufferInfo.setQueueFamilyIndices(queueFamilyIndices);
		}
		else {
			bufferInfo.sharingMode = vk::SharingMode::eExclusive;
		}

		buffer = m_Device.createBuffer(bufferInfo);

		const vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

		vk::MemoryAllocateInfo allocInfo;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		bufferMemory = m_Device.allocateMemory(allocInfo);

		buffer.bindMemory(bufferMemory, 0);
	}

	void createImage(
		const uint32_t width,
		const uint32_t height,
		const uint32_t mipLevels,
		const vk::SampleCountFlagBits numSamples,
		const vk::Format format,
		const vk::ImageTiling tiling,
		const vk::ImageUsageFlags usage,
		const vk::MemoryPropertyFlags properties,
		vk::raii::Image& image,
		vk::raii::DeviceMemory& imageMemory
	) const
	{
		vk::ImageCreateInfo imageInfo;
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = vk::ImageLayout::eUndefined;
		imageInfo.usage = usage;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;
		imageInfo.samples = numSamples;
		image = m_Device.createImage(imageInfo);
		const vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
		imageMemory = m_Device.allocateMemory(allocInfo);
		image.bindMemory(*imageMemory, 0);
	}

	vk::raii::ImageView createImageView(
		const vk::Image image,
		const vk::Format format,
		const vk::ImageAspectFlags aspectFlags,
		const uint32_t mipLevels
	) const
	{
		vk::ImageViewCreateInfo viewInfo;
		viewInfo.image = image;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		vk::ImageSubresourceRange range;
		range.aspectMask = vk::ImageAspectFlagBits::eColor;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;
		range.aspectMask = aspectFlags;
		range.levelCount = mipLevels;
		viewInfo.setSubresourceRange(range);
		return m_Device.createImageView(viewInfo);
	}

	void copyBuffer(const vk::raii::Buffer& srcBuffer, const vk::raii::Buffer& dstBuffer, const vk::DeviceSize size) const
	{
		const vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::BufferCopy copyRegion;
		copyRegion.size = size;
		commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBufferToImage(
		const vk::raii::Buffer& buffer,
		const vk::raii::Image& image,
		const uint32_t width,
		const uint32_t height
	) const 
	{
		const vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::BufferImageCopy region;
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = vk::Offset3D{ 0, 0, 0 };
		region.imageExtent = vk::Extent3D{ width, height, 1 };

		commandBuffer.copyBufferToImage(
			buffer,
			image,
			vk::ImageLayout::eTransferDstOptimal,
			region
		);

		endSingleTimeCommands(commandBuffer);
	}

	vk::raii::CommandBuffer beginSingleTimeCommands() const {
		vk::CommandBufferAllocateInfo allocInfo;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		//allocInfo.commandPool = m_TransferCommandPool;
		allocInfo.commandPool = m_CommandPool;
		allocInfo.commandBufferCount = 1;

		auto commandBuffers = m_Device.allocateCommandBuffers(allocInfo);
		vk::raii::CommandBuffer commandBuffer = std::move(commandBuffers.at(0));

		vk::CommandBufferBeginInfo beginInfo;
		beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

		commandBuffer.begin(beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const {
		commandBuffer.end();

		vk::SubmitInfo submitInfo;
		submitInfo.setCommandBuffers(*commandBuffer);

		//m_TransferQueue.submit(submitInfo);
		//m_TransferQueue.waitIdle();
		m_GraphicsQueue.submit(submitInfo);
		m_GraphicsQueue.waitIdle();
	}

	void transitionImageLayout(
		const vk::raii::Image& image,
		const vk::Format format,
		const vk::ImageLayout oldLayout,
		const vk::ImageLayout newLayout,
		const uint32_t mipLevels
	) const
	{
		const vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::ImageMemoryBarrier barrier;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
		barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;

		barrier.image = image;
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		barrier.srcAccessMask = {}; // TODO
		barrier.dstAccessMask = {}; // TODO

		vk::PipelineStageFlagBits sourceStage;
		vk::PipelineStageFlagBits destinationStage;

		if (oldLayout == vk::ImageLayout::eUndefined &&
			newLayout == vk::ImageLayout::eTransferDstOptimal
			) {
			barrier.srcAccessMask = {};
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;
		}
		else if (
			oldLayout == vk::ImageLayout::eTransferDstOptimal &&
			newLayout == vk::ImageLayout::eShaderReadOnlyOptimal
			) {
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		commandBuffer.pipelineBarrier(
			sourceStage,
			destinationStage,
			{},     // dependencyFlags
			nullptr,    // memoryBarriers
			nullptr,    // bufferMemoryBarriers
			barrier     // imageMemoryBarriers
		);

		endSingleTimeCommands(commandBuffer);
	}

	vk::Format findDepthFormat(const std::vector<vk::Format>& candidates) const {
		for (const vk::Format format : candidates) {
			// vk::FormatProperties
			const auto props = m_PhysicalDevice.getFormatProperties(format);
			if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
				return format;
			}
		}
		throw std::runtime_error("failed to find supported format!");
	}

	void generateMipmaps(
		vk::raii::Image& image,
		vk::Format imageFormat,
		int32_t texWidth,
		int32_t texHeight,
		uint32_t mipLevels
	)
	{
		// vk::FormatProperties
		if (const auto formatProperties = m_PhysicalDevice.getFormatProperties(imageFormat);
			!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)
		) throw std::runtime_error("texture image format does not support linear blitting!");

		const auto commandBuffer = beginSingleTimeCommands();

		vk::ImageMemoryBarrier imageBarrier;
		imageBarrier.image = image;
		imageBarrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
		imageBarrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
		imageBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		imageBarrier.subresourceRange.baseArrayLayer = 0;
		imageBarrier.subresourceRange.layerCount = 1;
		imageBarrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;
		for (uint32_t i = 1; i < mipLevels; i++) {
			// 1. 将 i - 1 级别的图像布局转换成 eTransferSrcOptimal ，以便作为传输源
			// 2. 从 i - 1 级别中 blit 新图像到 i 级别
			// 3. 将 i - 1 级别的图像布局转换回 eShaderReadOnlyOptimal ，以便后面采样器读取
			imageBarrier.subresourceRange.baseMipLevel = i - 1;
			imageBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
			imageBarrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
			imageBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			imageBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
			commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eTransfer,
				{},
				nullptr,
				nullptr,
				imageBarrier
			);

			// 设置2D图像放缩范围 
			// 左上角 [0, 0, 0] ---> [0, 0, 0]
			// 右下角 [width, height, 1] ---> [width/2, height/2, 1]
			// 实际上是一个3D cube，z值的范围一直为[0, 1)
			vk::ImageBlit imageBlit;
			imageBlit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
			imageBlit.srcOffsets[1] = vk::Offset3D{ mipWidth, mipHeight, 1 };
			imageBlit.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
			imageBlit.dstOffsets[1] = vk::Offset3D{ 
				mipWidth > 1 ? mipWidth / 2 : 1,
				mipHeight > 1 ? mipHeight / 2 : 1,
				1 
			};
			imageBlit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageBlit.srcSubresource.mipLevel = i - 1;
			imageBlit.srcSubresource.baseArrayLayer = 0;
			imageBlit.srcSubresource.layerCount = 1;
			imageBlit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageBlit.dstSubresource.mipLevel = i;
			imageBlit.dstSubresource.baseArrayLayer = 0;
			imageBlit.dstSubresource.layerCount = 1;
			commandBuffer.blitImage(
				image, vk::ImageLayout::eTransferSrcOptimal,
				image, vk::ImageLayout::eTransferDstOptimal,
				imageBlit,
				vk::Filter::eLinear
			);

			imageBarrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
			imageBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			imageBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
			imageBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eFragmentShader,
				{},
				nullptr,
				nullptr,
				imageBarrier
			);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}
		imageBarrier.subresourceRange.baseMipLevel = mipLevels - 1;
		imageBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		imageBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		imageBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
		imageBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader,
			{},
			nullptr,
			nullptr,
			imageBarrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	vk::SampleCountFlagBits getMaxUsableSampleCount() {
		const auto properties = m_PhysicalDevice.getProperties();
		const vk::SampleCountFlags counts = (
			properties.limits.framebufferColorSampleCounts &
			properties.limits.framebufferDepthSampleCounts
		);

		if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
		if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
		if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
		if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
		if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
		if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;
		return vk::SampleCountFlagBits::e1;
	}

	void updateDynamicUniformBuffer(const uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();
		const auto currentTime = std::chrono::high_resolution_clock::now();
		const float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
		startTime = currentTime;

		m_DynamicUboMatrices[1] = glm::rotate(
			m_DynamicUboMatrices[1],    // 在原先的基础上旋转
			glm::radians(time * 60.0f),
			glm::vec3(0.0f, 1.0f, 0.0f)
		);

		memcpy(
			m_DynamicUniformBuffersMapped[currentImage],
			m_DynamicUboMatrices.data(),
			sizeof(glm::mat4) * m_DynamicUboMatrices.size()
		);
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
		for (const auto& it : m_UniformBuffersMemory)
		{
			it.unmapMemory();
		}

		for (const auto& it : m_DynamicUniformBuffersMemory)
		{
			it.unmapMemory();
		}

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

		updateUniformBuffer(m_CurrentFrame);
		updateDynamicUniformBuffer(m_CurrentFrame);

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

	void updateUniformBuffer(const uint32_t currentImage)
	{
		static auto lastTime = std::chrono::high_resolution_clock::now();
		const auto now = std::chrono::high_resolution_clock::now();
		const float deltaime = std::chrono::duration<float, std::chrono::seconds::period>(now - lastTime).count();
		lastTime = now;

		m_Camera.OnUpdate(deltaime);

		UniformBufferObject ubo{};
		ubo.view = m_Camera.GetView();
		ubo.proj = m_Camera.GetProjection();
		ubo.proj[1][1] *= -1.0f; // GLM is for OpenGL whose y-axis is bottom-to-up, but Vulkan is up-to-bottom

		memcpy(m_UniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
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