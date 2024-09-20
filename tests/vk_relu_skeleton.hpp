#ifndef __VK_RELU_SKELETON__
#define __VK_RELU_SKELETON__

#include <vulkan/vulkan.h>

struct Context
{
	uint32_t vk_loader_version;
	VkApplicationInfo app_info;
	VkInstance instance;
	VkPhysicalDevice phys_device;
	VkDevice device;  // logical device

	uint32_t queue_family_idx;
};

void infer_vk_loader_version(Context& ctx);
void print_vk_loader_version(Context& ctx);

void initate_app_info(Context& ctx);
void create_instance(Context& ctx);
void select_physical_device(Context& ctx);
void create_logical_device(Context& ctx);


void destroy_context(Context& ctx);

/// @brief testing the current vulkan impl.
void run_vk_compute();

#endif  // __VK_RELU_SKELETON__
