#include "gemma2_tests.hpp"
#include "test_tools.hpp"
#include "dat_file.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <filesystem>

#include "gemma_mlp.hpp"
#include "gemma_decoder.hpp"
#include "gemma_model.hpp"
#include "zamba_rotary.hpp"


const std::filesystem::path artifact_folder_path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\xgemma2_tests";


void external_test_gemma2_decoder_mlp()
{
	auto path = artifact_folder_path / "test_gemma2decoder_mlp";

	// read tensors from files
	auto hx = load_tensor((path / "in_0.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	GemmaMLPweights<float32, CUDA> mlp_weights;
	mlp_weights.load_weights(
		(path / "gemma2decoder_mlp.gemma2mlp.gate_proj.weight.dat").string(),
		(path / "gemma2decoder_mlp.gemma2mlp.up_proj.weight.dat").string(),
		(path / "gemma2decoder_mlp.gemma2mlp.down_proj.weight.dat").string()
	);

	auto dx = hx.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_mlp(mlp_weights, dx);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_gemma2_decoder_mlp - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_decoder_rmsn()
{
	auto path = artifact_folder_path / "test_gemma2decoder_inprmsn";

	// read tensors from files
	auto hx = load_tensor((path / "in_0.dat").string());
	auto hw = load_tensor((path / "gemma2decoder_inprmsn.gemma2rmsnorm.weight.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_rms_norm(dx, -1, dw, 1e-6f, true);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_gemma2_decoder_rmsn - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_decoder_attention()
{
	auto path = artifact_folder_path / "test_gemma2decoder_attention";

	// read tensors from files
	auto h_hs = load_tensor((path / "in_hidden_states.dat").string());
	auto h_atten_mask = load_tensor((path / "in_attention_mask.dat").string());
	auto h_pos_ids = load_tensor<int32>((path / "in_position_ids.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	GemmaSDPAweights<float32> sdpa_weights;
	sdpa_weights.load_weights(
		(path / "gemma2decoder_attention.gemma2sdpaattention.q_proj.weight.dat").string(),
		(path / "gemma2decoder_attention.gemma2sdpaattention.k_proj.weight.dat").string(),
		(path / "gemma2decoder_attention.gemma2sdpaattention.v_proj.weight.dat").string(),
		(path / "gemma2decoder_attention.gemma2sdpaattention.o_proj.weight.dat").string()
	);

	GemmaKVcache kv_cache = {};

	auto d_hs = h_hs.copy_to_cuda();
	auto d_atten_mask = h_atten_mask.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_sdpa(sdpa_weights, kv_cache, d_hs, d_atten_mask, d_pos_ids, 256, 10000, 0.0625f);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_gemma2_decoder_attention - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_attention_rotary()
{
	auto path = artifact_folder_path / "test_gemma2attention_calcrotary";

	// read tensors from files
	auto hq = load_tensor((path / "in_0.dat").string());
	auto hk = load_tensor((path / "in_1.dat").string());
	auto h_pos_ids = load_tensor<int32>((path / "in_3.dat").string());
	auto exp_hq = load_tensor((path / "out_0.dat").string());
	auto exp_hk = load_tensor((path / "out_1.dat").string());

	auto dq = hq.copy_to_cuda();
	auto dk = hk.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();

	// the zamba is compatible with gemma2!
	// alt rotary expects different axes order (seq should be on the second place)
	auto freq = tensor_zamba_precomp_rotary_embedding<float32>(d_pos_ids, 256);
	auto act_dq_cuda = tensor_apply_zamba_rotary_embedding(dq, freq);
	auto act_dk_cuda = tensor_apply_zamba_rotary_embedding(dk, freq);

	auto act_hq_cuda = act_dq_cuda.copy_to_host();
	auto act_hk_cuda = act_dk_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = /*cmp(exp_hq, act_hq_cuda); &&*/ cmp(exp_hk, act_hk_cuda);
	std::cout << "TestCase [external_test_gemma2_attention_rotary - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_model_decoder()
{
	auto path = artifact_folder_path / "test_gemma2model_decoder";

	// read tensors from files
	auto h_hidden_states = load_tensor((path / "in_0.dat").string());
	auto h_attn_mask = load_tensor((path / "in_attention_mask.dat").string());
	auto h_pos_ids = load_tensor<int32>((path / "in_position_ids.dat").string());

	auto exp_hy = load_tensor((path / "out_0.dat").string());

	GemmaKVcache kv_cache = {};

	GemmaDecoderweights<float32> gd_weights;
	gd_weights.load_weights(
		(path / "gemm2model_decoder.gemma2decoderlayer.input_layernorm.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.pre_feedforward_layernorm.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.post_feedforward_layernorm.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.post_attention_layernorm.weight.dat").string(),

		(path / "gemm2model_decoder.gemma2decoderlayer.self_attn.q_proj.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.self_attn.k_proj.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.self_attn.v_proj.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.self_attn.o_proj.weight.dat").string(),

		(path / "gemm2model_decoder.gemma2decoderlayer.mlp.gate_proj.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.mlp.up_proj.weight.dat").string(),
		(path / "gemm2model_decoder.gemma2decoderlayer.mlp.down_proj.weight.dat").string()
	);

	const int hdim = 256;
	const int rope_base = 10000;
	float32 rms_norm_eps = 1e-6f;
	float32 sfmx_scale = 0.0625f;

	auto d_hidden_states = h_hidden_states.copy_to_cuda();
	auto d_attn_mask = h_attn_mask.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();

	auto act_dy_cuda = tensor_gemma_decoder(
		gd_weights,
		kv_cache,
		d_hidden_states,
		d_attn_mask,
		d_pos_ids,
		hdim,
		rope_base,
		rms_norm_eps,
		sfmx_scale
	);

	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_gemma2_model_decoder - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}

