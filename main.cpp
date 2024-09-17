#include "tests.hpp"
#include "performance.hpp"
#include "safetensors_file.hpp"


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

	test_sdp_fwd_f32();

	test_sdp_bwd_f32();

	test_softmax_bwd_f32();

	external_test_sdp_fwd_f32();*/

	//external_test_cpu_softmax_bwd_f32();

	/*test_quant_lin_f32_i8();
	test_dequant_lin_i8_f32();
	test_qmm_i8_f32();

	test_quant_sdp_fwd_f32_i8();*/

	//external_test_sf_data_reading();

	/* performance measurement (cuda) */

	test_mm_f32_640x1280_1280x320();
	
	//perf_mm_f16_640x1280_1280x320();


	/*std::string path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\safetensors\\diffusion_pytorch_model.safetensors";
	std::vector<Tensor<float32, CPU>> tensors;
	sft_read_tensors(path, tensors);

	std::cout << represent_tensor(tensors[0]) << "\n";
	std::cout << represent_tensor(tensors[1]) << "\n";*/

	std::cout << "Finished" << std::endl;

	return 0;
}
