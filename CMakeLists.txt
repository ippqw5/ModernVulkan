# CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0)

if (MSVC)
    # add /utf-8 option
    add_compile_options(/utf-8)
endif()

project(HelloVulkan LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 此处使用 23 标准，你可以按需设置
set(CMAKE_CXX_STANDARD 23)

# 需要设置环境变量VULKAN_SDK，比如 D:\Vulkan\1.4.309.0
# 环境变量默认在Vulkan SDK安装时，自动设置
find_package(Vulkan REQUIRED)

# 添加可执行程序目标
add_executable(${PROJECT_NAME} 
    "src/main.cpp"
    "src/camera.cpp"
    "src/input.cpp"
    "src/application.cpp"
)

# 设置thirdparty路径
set(THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/thirdparty)
    
# 添加包含目录
target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR}/include/stb)
target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR}/include/tinyobjloader)

# 链接Vulkan
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
    
# 链接GLFW库
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR}/lib/glfw/glfw3.lib)
    # Add necessary Windows libraries for GLFW
    target_link_libraries(${PROJECT_NAME} PRIVATE user32 gdi32 shell32)
else()
    # For Linux/Mac, you might need different library names
    target_link_libraries(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR}/lib/glfw/libglfw3.a)
endif()
    
# GLM is header-only, so no linking needed

# STB is header-only, so no linking needed

# tinyobjloader is header-only, so no linking needed

add_subdirectory(shaders)

# Ensure shaders are built first and copied next to the executable
add_dependencies(${PROJECT_NAME} CompileShaders)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/shaders"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders")

# add `textures` folder to working directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/textures $<TARGET_FILE_DIR:${PROJECT_NAME}>/textures)

# add `models` folder to working directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:${PROJECT_NAME}>/models)
