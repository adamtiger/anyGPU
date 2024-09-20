#include "vk_relu_skeleton.hpp"

#include <iostream>
#include <sstream>
#include <vector>


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
	// query the physical devices
	uint32_t num_phys_devices;
	CHECK_RESULT(vkEnumeratePhysicalDevices(ctx.instance, &num_phys_devices, 0));

	std::vector<VkPhysicalDevice> phys_devices(num_phys_devices);
	CHECK_RESULT(vkEnumeratePhysicalDevices(ctx.instance, &num_phys_devices, phys_devices.data()));

	// select the right physical device
	for (auto phys_dev : phys_devices)
	{
		VkPhysicalDeviceProperties phys_dev_props;
		vkGetPhysicalDeviceProperties(phys_dev, &phys_dev_props);

		if (phys_dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)  // TODO: add more checks
		{
			// save selected device
			ctx.phys_device = phys_dev;

			// print selected device data
			std::cout << "Selected device name: " << phys_dev_props.deviceName << std::endl;
			std::cout << "  Vendor id: " << phys_dev_props.vendorID << std::endl;

			return;
		}
	}

	std::cout << "Error: no device was found \n";
	exit(1);
}

void create_logical_device(Context& ctx)
{
	// select an appropriate queue family
	uint32_t queue_family_prop_count;
	vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys_device, &queue_family_prop_count, 0);

	std::vector<VkQueueFamilyProperties> queue_family_props(queue_family_prop_count);
	vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys_device, &queue_family_prop_count, queue_family_props.data());

	uint32_t qidx = 0;
	for (auto& queue_family_prop : queue_family_props)
	{
		bool at_least_one = (queue_family_prop.queueCount > 0);  // guaranteed
		bool has_compute_bit = queue_family_prop.queueFlags & VK_QUEUE_COMPUTE_BIT;
		bool has_transfer_bit = queue_family_prop.queueFlags & VK_QUEUE_TRANSFER_BIT;

		if (at_least_one && has_compute_bit && has_transfer_bit)
		{
			ctx.queue_family_idx = qidx;
			break;
		}

		qidx += 1;
	}

	// TODO: report error if not found

	// create the queues for the logical device
	VkDeviceQueueCreateInfo queue_crt_info = {};
	queue_crt_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queue_crt_info.flags = VK_QUEUE_COMPUTE_BIT;
	queue_crt_info.queueCount = 1;
	queue_crt_info.queueFamilyIndex = ctx.queue_family_idx;

	// create the logical device
	VkDeviceCreateInfo logical_device_crt_info = {};
	logical_device_crt_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	logical_device_crt_info.queueCreateInfoCount = 1;
	logical_device_crt_info.pQueueCreateInfos = &queue_crt_info;

	CHECK_RESULT(vkCreateDevice(ctx.phys_device, &logical_device_crt_info, 0, &ctx.device));
}




void destroy_context(Context& ctx)
{
	vkDestroyDevice(ctx.device, 0);
	
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
	select_physical_device(ctx);
	create_logical_device(ctx);



	// clean up
	destroy_context(ctx);
}
