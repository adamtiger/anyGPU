#include "tests.hpp"

int main()
{
	//std::cout << print_cuda_device_props();

	//test_binary_add_f32();
	//test_binary_add_i32();
	test_mm_f32();

	// experiment
	/*auto ta = crt_random_tensor<int32, CUDA>({ 5, 300 }, 11);
	auto tb = crt_random_tensor<int32, CUDA>({ 5, 300 }, 18);

	auto tc = tensor_add(ta, tb);

	std::cout << represent_tensor(ta.copy_to_host()) << std::endl;
	std::cout << represent_tensor(tb.copy_to_host()) << std::endl;
	std::cout << represent_tensor(tc.copy_to_host()) << std::endl;*/

	return 0;
}
