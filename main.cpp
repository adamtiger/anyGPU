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

	test_softmax_f32();

	test_sdp_fwd_f32();*/ 

	// test_sdp_bwd_f32();

	// test_softmax_bwd_f32();

	//external_test_sdp_fwd_f32();

	//external_test_cpu_softmax_bwd_f32();

	test_quant_lin_f32_i8();
	test_dequant_lin_i8_f32();
	test_qmm_i8_f32();

	test_quant_sdp_fwd_f32_i8();

	// experiment
	/*auto qw = crt_random_tensor<float32, CUDA>({ 16, 64 }, 11);
	auto kw = crt_random_tensor<float32, CUDA>({ 16, 64 }, 15);
	auto vw = crt_random_tensor<float32, CUDA>({ 16, 64 }, 18);

	auto y = single_head_attention_fwd<float32, CUDA, NONE, FULL>(qw, kw, vw);

	std::cout << represent_tensor(y.copy_to_host(), 15) << std::endl;*/

	/*auto q = load_tensor("C:\\Data\\AI\\projects\\anyGPU\\artifacts\\sdp_nomask_noscore_f32_16_64\\q.dat");
	auto y = load_tensor("C:\\Data\\AI\\projects\\anyGPU\\artifacts\\sdp_nomask_noscore_f32_16_64\\y.dat");

	std::cout << represent_tensor(q, 10) << std::endl;
	std::cout << represent_tensor(y, 10) << std::endl;*/

	return 0;
}
