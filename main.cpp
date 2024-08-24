#include "tests.hpp"

#include "softmax_ops.hpp"

int main()
{
	//std::cout << print_cuda_device_props();

	test_binary_add_f32();
	test_binary_add_i32();

	test_binary_mul_f32();
	test_binary_mul_i32();


	test_mm_f32();

	test_transp_f32();

	test_softmax_f32();

	// experiment
	/*auto ta = crt_random_tensor<float32, CPU>({ 5, 3 }, 11);

	auto tc = tensor_softmax(ta);

	std::cout << represent_tensor(ta, 15) << std::endl;
	std::cout << represent_tensor(tc, 15) << std::endl;*/

	return 0;
}
