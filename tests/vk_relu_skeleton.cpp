#include "vk_relu_skeleton.hpp"

#include <iostream>
#include <sstream>


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
	bool found = false;
	for (auto& queue_family_prop : queue_family_props)
	{
		bool at_least_one = (queue_family_prop.queueCount > 0);  // guaranteed
		bool has_compute_bit = queue_family_prop.queueFlags & VK_QUEUE_COMPUTE_BIT;
		bool has_transfer_bit = queue_family_prop.queueFlags & VK_QUEUE_TRANSFER_BIT;

		if (at_least_one && has_compute_bit && has_transfer_bit)
		{
			ctx.queue_family_idx = qidx;
			found = true;
			break;
		}

		qidx += 1;
	}

	// report error if not found
	if (!found)
	{
		std::cout << "Error: no suitable queue family was found \n";
		exit(1);
	}

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

VTensor create_tensor(const Context& ctx, const std::vector<int>& shape)
{
	// calculate the required buffer size (bytes)
	int buffer_size = 1;
	for (int s : shape)
	{
		buffer_size *= s;
	}

	buffer_size *= sizeof(float);

	// create the tensor
	VTensor tensor = {};
	tensor.shape = shape;
	tensor.buffer_size = buffer_size;

	VkBufferCreateInfo buffer_ci = {};
	buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_ci.size = buffer_size;

	vkCreateBuffer(ctx.device, &buffer_ci, 0, &tensor.buffer);

	return tensor;
}

void calculate_relu(const Context& ctx, const VTensor& x, VTensor& y)
{
	// create descriptor sets and accompanying structures

	VkDescriptorSetLayoutBinding ds_layout_bind_x = {};
	ds_layout_bind_x.binding = 0;
	ds_layout_bind_x.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	ds_layout_bind_x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ds_layout_bind_x.descriptorCount = 1;

	VkDescriptorSetLayoutBinding ds_layout_bind_y = {};
	ds_layout_bind_y.binding = 1;
	ds_layout_bind_y.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	ds_layout_bind_y.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ds_layout_bind_y.descriptorCount = 1;

	std::vector<VkDescriptorSetLayoutBinding> ds_layout_binds = { 
		ds_layout_bind_x, 
		ds_layout_bind_y 
	};


	VkDescriptorSetLayoutCreateInfo ds_layout_info = {};
	ds_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	ds_layout_info.bindingCount = 2;  // two tensors
	ds_layout_info.pBindings = ds_layout_binds.data();

	VkDescriptorSetLayout descr_set_layout;
	vkCreateDescriptorSetLayout(
		ctx.device,
		&ds_layout_info,
		0,
		&descr_set_layout
	);


	VkDescriptorPoolSize descr_pool_size = {};
	descr_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descr_pool_size.descriptorCount = 2;

	VkDescriptorPoolCreateInfo descr_pool_crt_info = {};
	descr_pool_crt_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descr_pool_crt_info.maxSets = 2;
	descr_pool_crt_info.poolSizeCount = 1;
	descr_pool_crt_info.pPoolSizes = &descr_pool_size;

	VkDescriptorPool descr_pool;
	vkCreateDescriptorPool(ctx.device, &descr_pool_crt_info, 0, &descr_pool);


	VkDescriptorSetAllocateInfo descr_set_alloc_info = {};
	descr_set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descr_set_alloc_info.descriptorPool = descr_pool;
	descr_set_alloc_info.descriptorSetCount = 1;
	descr_set_alloc_info.pSetLayouts = &descr_set_layout;
	

	VkDescriptorSet descr_set;
	vkAllocateDescriptorSets(ctx.device, &descr_set_alloc_info, &descr_set);


	VkWriteDescriptorSet wr_descr_set_x = {};
	wr_descr_set_x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	wr_descr_set_x.dstBinding = 0;
	wr_descr_set_x.descriptorCount = 1;
	wr_descr_set_x.dstSet = descr_set;
	wr_descr_set_x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	VkDescriptorBufferInfo buf_info_x = {};
	{
		buf_info_x.buffer = x.buffer;
		buf_info_x.range = x.buffer_size;
		buf_info_x.offset = 0;
	}
	wr_descr_set_x.pBufferInfo = &buf_info_x;

	VkWriteDescriptorSet wr_descr_set_y = {};
	wr_descr_set_y.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	wr_descr_set_y.dstBinding = 1;
	wr_descr_set_y.descriptorCount = 1;
	wr_descr_set_y.dstSet = descr_set;
	wr_descr_set_y.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	VkDescriptorBufferInfo buf_info_y = {};
	{
		buf_info_y.buffer = y.buffer;
		buf_info_y.range = y.buffer_size;
		buf_info_y.offset = 0;
	}
	wr_descr_set_y.pBufferInfo = &buf_info_y;

	std::vector<VkWriteDescriptorSet> wr_descr_sets = {
		wr_descr_set_x,
		wr_descr_set_y
	};


	vkUpdateDescriptorSets(ctx.device, 2, wr_descr_sets.data(), 0, 0);



	// create compute pipeline

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
