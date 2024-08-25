#include "tests.hpp"

#include "attention.hpp"

int main()
{
	//std::cout << print_cuda_device_props();

	/*test_binary_add_f32();
	test_binary_add_i32();

	test_binary_mul_f32();
	test_binary_mul_i32();


	test_mm_f32();

	test_transp_f32();

	test_softmax_f32();*/

	test_sdp_fwd_f32();

	// experiment
	/*auto qw = crt_random_tensor<float32, CUDA>({ 16, 64 }, 11);
	auto kw = crt_random_tensor<float32, CUDA>({ 16, 64 }, 15);
	auto vw = crt_random_tensor<float32, CUDA>({ 16, 64 }, 18);

	auto y = single_head_attention_fwd<float32, CUDA, NONE, FULL>(qw, kw, vw);

	std::cout << represent_tensor(y.copy_to_host(), 15) << std::endl;*/

	return 0;
}
