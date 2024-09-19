#include "vk_relu_skeleton.hpp"

#include <sstream>
#include <iostream>


// helper function to get the error immediately
//  it halts the program execution
static void CHECK_RESULT(VkResult result)
{
	if (result != VK_SUCCESS)
	{
		std::cout << "Failed vk operation: " << result << "\n";
		std::cout << "Exiting ..." << std::endl;
		exit(1);
	}
}

// loader info

void infer_vk_loader_version(Context& state)
{
	CHECK_RESULT(vkEnumerateInstanceVersion(&state.vk_loader_version));
}

void print_vk_loader_version(Context& state)
{
	uint32_t api_version = state.vk_loader_version;
	unsigned int variant = (api_version >> 29u);
	unsigned int major_ver = (api_version >> 22u) & 0x007Fu;
	unsigned int minor_ver = (api_version >> 12u) & 0x03FFu;
	unsigned int patch_ver = (api_version & 0x0FFFu);

	std::stringstream ss;
	ss << "VK loader version: ";
	ss << major_ver;
	ss << ".";
	ss << minor_ver;
	ss << ".";
	ss << patch_ver;

	if (variant != 0)
	{
		ss << ".";
		ss << variant;
	}

	std::cout << ss.str() << std::endl;
}


// implementation

void initate_app_info(Context& ctx)
{
	ctx.app_info = {};
	ctx.app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	ctx.app_info.pNext = NULL;
	ctx.app_info.apiVersion = VK_API_VERSION_1_0;
	ctx.app_info.applicationVersion = (1 << 22u | 0 << 12u | 0);
	ctx.app_info.pApplicationName = "anygpu";
}

void create_instance(Context& ctx)
{
	VkInstanceCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	info.pApplicationInfo = &ctx.app_info;
	info.pNext = NULL;
	info.flags = 0;
	info.enabledLayerCount = 0;
	info.enabledExtensionCount = 0;

	CHECK_RESULT(vkCreateInstance(&info, nullptr, &ctx.instance));
}

void select_physical_device(Context& ctx)
{
	
}


// main
void run_vk_compute()
{
	Context ctx;

	// verions
	infer_vk_loader_version(ctx);
	print_vk_loader_version(ctx);

	// devices
	initate_app_info(ctx);
	create_instance(ctx);
}
