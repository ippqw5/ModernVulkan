#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    uint enableTexture;
} pc; //因为推送常量只有一个数据区，直接使用 push_constant 标记即可。 PushConstants 是自定义的类型名。

void main() {
    if (pc.enableTexture > 0) {
        outColor = texture(texSampler, fragTexCoord);
    } else {
        outColor = vec4(fragColor, 1.0);
    }
}