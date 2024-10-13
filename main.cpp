#include "tests.hpp"
#include "performance.hpp"
#include "safetensors_file.hpp"
#include "vk_relu_skeleton.hpp"

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

	external_test_sdp_fwd_f32();

	external_test_cpu_softmax_bwd_f32();

	test_quant_lin_f32_i8();
	test_dequant_lin_i8_f32();
	test_qmm_i8_f32();

	test_quant_sdp_fwd_f32_i8();

	external_test_sf_data_reading();

	external_test_layer_norm_fwd_f32();
	external_test_rms_norm_fwd_f32();*/
	//external_test_silu_fwd_f32();
	//external_test_gelu_fwd_f32();
	//external_test_gelu_approx_fwd_f32();
	//external_test_embedding_fwd_f32();
	//external_test_rotary_embedding_fwd_f32();
	//external_test_alt_rotary_embedding_fwd_f32();
	//external_test_linear_fwd_f32();
	//external_test_transpose_fwd_f32();
	//external_test_sdp_masked_scaled_fwd_f32();
	//external_test_concat_fwd_f32();
	//external_test_repeat_fwd_f32();
	//external_test_causal_conv1d_fwd_f32();

	//external_test_zamba2_model_rmsnorm();
	//external_test_zamba2_attndeco_attn();
	//external_test_zamba2_attndeco_mlp();
	//external_test_zamba2_attn_rotary();
	//external_test_zamba2_attn_sdp();
	//external_test_zamba2_model_attndecoder();
	//external_test_mamba2_layer_causal_conv1d();
	//external_test_mamba2_layer_gated_rmsnorm();
	//test_zamba2_glu();

	//external_test_gemma2_decoder_mlp();
	//external_test_gemma2_decoder_rmsn
	external_test_gemma2_decoder_attention();
	//external_test_gemma2_attention_rotary();
	//external_test_gemma2_model_decoder();

	//run_vk_compute();

	/* performance measurement (cuda) */

	// test_mm_f32_640x1280_1280x320();
	
	//perf_mm_f16_640x1280_1280x320();


	/*std::string path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\safetensors\\diffusion_pytorch_model.safetensors";
	std::vector<Tensor<float32, CPU>> tensors;
	sft_read_tensors(path, tensors);

	std::cout << represent_tensor(tensors[0]) << "\n";
	std::cout << represent_tensor(tensors[1]) << "\n";*/

	std::cout << "Finished" << std::endl;

	return 0;
}
