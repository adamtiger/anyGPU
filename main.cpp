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

	external_test_sdp_fwd_f32();

	external_test_cpu_softmax_bwd_f32();

	test_quant_lin_f32_i8();
	test_dequant_lin_i8_f32();
	test_qmm_i8_f32();

	test_quant_sdp_fwd_f32_i8();

	external_test_sf_data_reading();

	external_test_layer_norm_fwd_f32();
	external_test_rms_norm_fwd_f32();*/
	/*external_test_silu_fwd_f32();
	external_test_gelu_fwd_f32();
	external_test_gelu_approx_fwd_f32();
	external_test_embedding_fwd_f32();
	external_test_rotary_embedding_fwd_f32();
	external_test_alt_rotary_embedding_fwd_f32();
	external_test_linear_fwd_f32();
	external_test_transpose_fwd_f32();
	external_test_sdp_masked_scaled_fwd_f32();
	external_test_concat_fwd_f32();
	external_test_repeat_fwd_f32();
	external_test_slice_fwd_f32();*/
	//external_test_causal_conv1d_fwd_f32();

	//external_test_gemma2_decoder_mlp();
	//external_test_gemma2_decoder_rmsn();
	//external_test_gemma2_decoder_attention();
	//external_test_gemma2_attention_rotary();
	//external_test_gemma2_model_decoder();
	//external_test_gemma2_model_decoder_sl1();
	//external_test_gemma2_lmhead_softcap();
	//external_test_gemma2_update_mask();
	//external_test_gemma2_kvcache_update();
	//external_test_gemma2_model_decoder_15();
	//external_test_gemma2_model_decoder_16();
	//external_test_gemma2_slide_mask();
	//external_test_gemma2_causallm();

	//external_test_gemma2_decoder_fused_mlp();

	external_test_mm_m1024_n2048_k2304_f32();


	/*std::string path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\safetensors\\gemma2_2b\\model-00003-of-00003.safetensors";
	std::vector<Tensor<float32, CPU>> tensors;
	sft_read_tensors(path, tensors);*/

	/*std::cout << represent_tensor(tensors[0]) << "\n";
	std::cout << represent_tensor(tensors[1]) << "\n";*/

	std::cout << "Finished" << std::endl;

	return 0;
}
