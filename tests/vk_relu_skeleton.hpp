#ifndef __VK_RELU_SKELETON__
#define __VK_RELU_SKELETON__

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

struct Context
{
	uint32_t vk_loader_version;
	VkApplicationInfo app_info;
	VkInstance instance;
	VkPhysicalDevice phys_device;
	VkDevice device;  // logical device

	uint32_t queue_family_idx;
};

struct VTensor
{
	std::vector<int> shape;
	VkBuffer buffer;
	VkDeviceMemory memory;
	// dtype is float32
	int buffer_size;  // buffer size in bytes
};

void infer_vk_loader_version(Context& ctx);
void print_vk_loader_version(Context& ctx);

void initate_app_info(Context& ctx);
void create_instance(Context& ctx);
void select_physical_device(Context& ctx);
void create_logical_device(Context& ctx);

int findMemoryProperties(
	const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
	uint32_t memoryTypeBitsRequirement,
	VkMemoryPropertyFlags requiredProperties
);

std::vector<char> load_shader_file(
	const std::string& path_to_shader);

VTensor create_tensor(
	const Context& ctx, 
	const std::vector<int>& shape, 
	const std::vector<float>& data);

std::vector<float> copy_tensor_data_to_host(
	const Context& ctx, 
	const VTensor& tensor);

void calculate_relu(
	const Context& ctx, 
	const VTensor& x, 
	VTensor& y);

void destroy_context(Context& ctx);

/// @brief testing the current vulkan impl.
void run_vk_compute();

#endif  // __VK_RELU_SKELETON__
