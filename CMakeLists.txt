# CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0)

# USE_VCPKG variable is now controlled via CMakeSettings.json
# Default to OFF if not specified
if(NOT DEFINED USE_VCPKG)
    set(USE_VCPKG OFF)
endif()

# Set vcpkg toolchain file before project() if using vcpkg
if(USE_VCPKG)
    # 需要设置环境变量VCPKG_ROOT
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" VCPKG_CMAKE_PATH)
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_CMAKE_PATH}/scripts/buildsystems/vcpkg.cmake")
endif()

project(HelloVulkan LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 此处使用 23 标准，你可以按需设置
set(CMAKE_CXX_STANDARD 23)

# 需要设置环境变量VULKAN_SDK，比如 D:\Vulkan\1.4.309.0
# 环境变量默认在Vulkan SDK安装时，自动设置
find_package(Vulkan REQUIRED)

if(USE_VCPKG)
    message(STATUS "Using vcpkg for third-party libraries")
    
    # 通过vcpkg导入第三方库
    find_package(glfw3 CONFIG REQUIRED)
    find_package(glm CONFIG REQUIRED)
    find_package(Stb REQUIRED)
    
    # 添加可执行程序目标
    add_executable(${PROJECT_NAME} src/main.cpp)
    
    # 包含目录
    target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})

    # 链接库
    target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
    target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
    
else()
    message(STATUS "Using local thirdparty libraries")
    
    # 添加可执行程序目标
    add_executable(${PROJECT_NAME} src/main.cpp)
    
    # 设置thirdparty路径
    set(THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/thirdparty)
    
    # 添加包含目录
    target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR}/include/stb)
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
    
endif()

add_subdirectory(shaders)

# add `textures` folder to working directory?
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/textures $<TARGET_FILE_DIR:${PROJECT_NAME}>/textures)
