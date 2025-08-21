# CMakeLists.txt
cmake_minimum_required(VERSION 3.9.0)

# 需要设置环境变量VCPKG_ROOT
file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" VCPKG_CMAKE_PATH)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_CMAKE_PATH}/scripts/buildsystems/vcpkg.cmake")

project(HelloVulkan LANGUAGES CXX)

# 此处使用 23 标准，你可以按需设置
set(CMAKE_CXX_STANDARD 23)

# 需要设置VULKAN_SDK环境变量，比如 D:\Vulkan\1.4.309.0
# 环境变量默认在Vulkan SDK安装时，自动设置
find_package(Vulkan REQUIRED)

# 通过vcpkg导入第三方库
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# 添加可执行程序目标
add_executable(${PROJECT_NAME} src/main.cpp)

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan )
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm )
target_link_libraries(${PROJECT_NAME} PRIVATE glfw )