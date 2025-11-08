#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} uMVP;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 3) in mat4 inModel; //一个 location 最多 16 字节，所以 inModel 直接占了 4 个位置。
layout(location = 7) in uint inEnableTexture;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) flat out uint enableTexture; //flat 标记，阻止管线从顶点到片段着色器时对其进行查值操作，片段着色器会直接使用第一个顶点传递的值（通常）

void main() {
    gl_Position= uMVP.proj * uMVP.view * inModel * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}