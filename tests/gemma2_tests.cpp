#include "gemma2_tests.hpp"
#include "test_tools.hpp"
#include "dat_file.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <filesystem>

#include "gemma_mlp.hpp"
#include "gemma_decoder.hpp"
#include "gemma_model.hpp"
#include "gemma_causallm.hpp"
#include "gemma_linsoftcap.hpp"
#include "gemma_update_mask.hpp"
#include "gemma_kv_cache.hpp"
#include "gemma_slide_mask.hpp"
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
	auto h_cache_pos = load_tensor<int32>((path / "in_cache_position.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	GemmaSDPAweights<float32> sdpa_weights;
	sdpa_weights.load_weights(
		(path / "gemma2decoder_attention.gemma2sdpaattention.q_proj.weight.dat").string(),
		(path / "gemma2decoder_attention.gemma2sdpaattention.k_proj.weight.dat").string(),
		(path / "gemma2decoder_attention.gemma2sdpaattention.v_proj.weight.dat").string(),
		(path / "gemma2decoder_attention.gemma2sdpaattention.o_proj.weight.dat").string()
	);

	GemmaConfig config;

	// initiate kv cache
	GemmaKVcache kv_cache;
	kv_cache.init_cache(config, 1);

	auto d_hs = h_hs.copy_to_cuda();
	auto d_atten_mask = h_atten_mask.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();
	auto d_cache_pos = h_cache_pos.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_sdpa(
		config, 
		sdpa_weights, 
		kv_cache, 
		d_hs, 
		d_atten_mask, 
		d_pos_ids, 
		d_cache_pos,
		0, 
		config.sliding_window);
	
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
	auto h_cache_pos = load_tensor<int32>((path / "in_cache_position.dat").string());

	auto exp_hy = load_tensor((path / "out_0.dat").string());

	GemmaConfig config;

	// initiate kv cache
	GemmaKVcache kv_cache;
	kv_cache.init_cache(config, 1);

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

	auto d_hidden_states = h_hidden_states.copy_to_cuda();
	auto d_attn_mask = h_attn_mask.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();
	auto d_cache_pos = h_cache_pos.copy_to_cuda();

	auto act_dy_cuda = tensor_gemma_decoder(
		config,
		gd_weights,
		kv_cache,
		d_hidden_states,
		d_attn_mask,
		d_pos_ids,
		d_cache_pos,
		0,
		config.sliding_window
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


void external_test_gemma2_model_decoder_sl1()
{
	auto path = artifact_folder_path / "test_gemma2model_decoder_15";

	// read tensors from files
	auto h_hidden_states = load_tensor((path / "io_ckpt_4" / "in_0.dat").string());
	auto h_attn_mask = load_tensor((path / "io_ckpt_4" / "in_attention_mask.dat").string());
	auto h_pos_ids = load_tensor<int32>((path / "io_ckpt_4" / "in_position_ids.dat").string());
	auto h_cache_pos = load_tensor<int32>((path / "io_ckpt_4" / "in_cache_position.dat").string());

	auto exp_hy = load_tensor((path / "io_ckpt_4" / "out_0.dat").string());

	GemmaConfig config;

	// initiate kv cache
	GemmaKVcache kv_cache;
	kv_cache.init_cache(config, 1);

	GemmaDecoderweights<float32> gd_weights;
	gd_weights.load_weights(
		(path / "gemma2model_decoder_15.gemma2decoderlayer.input_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.pre_feedforward_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.post_feedforward_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.post_attention_layernorm.weight.dat").string(),

		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.q_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.k_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.v_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.o_proj.weight.dat").string(),

		(path / "gemma2model_decoder_15.gemma2decoderlayer.mlp.gate_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.mlp.up_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.mlp.down_proj.weight.dat").string()
	);

	auto d_hidden_states = h_hidden_states.copy_to_cuda();
	auto d_attn_mask = h_attn_mask.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();
	auto d_cache_pos = h_cache_pos.copy_to_cuda();

	auto act_dy_cuda = tensor_gemma_decoder(
		config,
		gd_weights,
		kv_cache,
		d_hidden_states,
		d_attn_mask,
		d_pos_ids,
		d_cache_pos,
		0,
		config.sliding_window
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
	std::cout << "TestCase [external_test_gemma2_model_decoder_sl1 - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_lmhead_softcap()
{
	auto path = artifact_folder_path / "test_gemma2_lmhead_softcap";

	// read tensors from files
	// hw is tied to the embedding weights
	auto hx = load_tensor((path / "in_0.dat").string());
	auto hw = load_tensor((path / "gemma2_causal_lm.gemma2forcausallm.model.embed_tokens.weight.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_linear_softcap(dx, dw, 30.f);
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
	std::cout << "TestCase [external_test_gemma2_lmhead_softcap - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_update_mask()
{
	auto path = artifact_folder_path / "test_gemma2model_update_mask";

	// read tensors from files
	// hw is tied to the embedding weights
	auto h_mask = load_tensor((path / "in_0.dat").string());
	auto h_inp = load_tensor((path / "in_1.dat").string());
	auto h_cpos = load_tensor<int32>((path / "in_2.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto d_mask = h_mask.copy_to_cuda();
	auto d_inp = h_inp.copy_to_cuda();
	auto d_cpos = h_cpos.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_update_mask(
		d_mask, d_inp, d_cpos, 41
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
	std::cout << "TestCase [external_test_gemma2_update_mask - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_kvcache_update()
{
	auto path = artifact_folder_path / "test_gemma2attention_kvupdate";

	GemmaConfig config;

	// initiate kv cache
	GemmaKVcache kv_cache;
	kv_cache.init_cache(config, 1);

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test the cache in each state
	bool correct = true;
	for (int ix = 1; ix < 101; ++ix)
	{
		std::string fn = "io_ckpt_" + std::to_string(ix);
		auto ckp_path = path / fn;

		// read tensors from files
	    // hw is tied to the embedding weights
		auto h_k = load_tensor((ckp_path / "in_0.dat").string());
		auto h_v = load_tensor((ckp_path / "in_1.dat").string());
		auto h_cpos = load_tensor<int32>((ckp_path / "in_4.dat").string());
		auto exp_hk = load_tensor((ckp_path / "out_0.dat").string());
		auto exp_hv = load_tensor((ckp_path / "out_1.dat").string());

		auto d_k = h_k.copy_to_cuda();
		auto d_v = h_v.copy_to_cuda();
		auto d_cpos = h_cpos.copy_to_cuda();
		
		int32 layer_idx = (ix - 1) % 26;
		int32 sliding_window = (layer_idx % 2 == 0 ? config.sliding_window : -1);

		Tensor<float32, CUDA> act_dk_cuda;
		Tensor<float32, CUDA> act_dv_cuda;
		kv_cache.update_cache(
			d_k, d_v, d_cpos, 
			layer_idx, 
			sliding_window,
			act_dk_cuda, 
			act_dv_cuda
		);

		auto act_hk_cuda = act_dk_cuda.copy_to_host();
		auto act_hv_cuda = act_dv_cuda.copy_to_host();

		// test cuda
		correct = correct && cmp(exp_hk, act_hk_cuda);
		correct = correct && cmp(exp_hv, act_hv_cuda);
	}

	std::cout << "TestCase [external_test_gemma2_kvcache_update - CUDA]: " << (correct ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_model_decoder_15()
{
	auto path = artifact_folder_path / "test_gemma2model_decoder_15";

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers_l2(actual, expected);
			return eq;
		};

	GemmaConfig config;

	// initiate kv cache
	GemmaKVcache kv_cache;
	kv_cache.init_cache(config, 1);

	GemmaDecoderweights<float32> gd_weights;
	gd_weights.load_weights(
		(path / "gemma2model_decoder_15.gemma2decoderlayer.input_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.pre_feedforward_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.post_feedforward_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.post_attention_layernorm.weight.dat").string(),

		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.q_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.k_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.v_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.self_attn.o_proj.weight.dat").string(),

		(path / "gemma2model_decoder_15.gemma2decoderlayer.mlp.gate_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.mlp.up_proj.weight.dat").string(),
		(path / "gemma2model_decoder_15.gemma2decoderlayer.mlp.down_proj.weight.dat").string()
	);

	int32 layer_idx = 15;
	int32 sliding_window = (layer_idx % 2 == 0 ? config.sliding_window : -1);

	// test the cache in each state
	bool correct = true;
	for (int ix = 3; ix < 31; ++ix)
	{
		std::string fn = "io_ckpt_" + std::to_string(ix);
		auto ckp_path = path / fn;

		// read tensors from files
		// read tensors from files
		auto h_hidden_states = load_tensor((ckp_path / "in_0.dat").string());
		auto h_attn_mask = load_tensor((ckp_path / "in_attention_mask.dat").string());
		auto h_pos_ids = load_tensor<int32>((ckp_path / "in_position_ids.dat").string());
		auto h_cache_pos = load_tensor<int32>((ckp_path / "in_cache_position.dat").string());

		auto exp_hy = load_tensor((ckp_path / "out_0.dat").string());

		auto d_hidden_states = h_hidden_states.copy_to_cuda();
		auto d_attn_mask = h_attn_mask.copy_to_cuda();
		auto d_pos_ids = h_pos_ids.copy_to_cuda();
		auto d_cache_pos = h_cache_pos.copy_to_cuda();

		auto act_dy_cuda = tensor_gemma_decoder(
			config,
			gd_weights,
			kv_cache,
			d_hidden_states,
			d_attn_mask,
			d_pos_ids,
			d_cache_pos,
			layer_idx,
			sliding_window
		);

		auto act_hy_cuda = act_dy_cuda.copy_to_host();

		// test cuda
		correct = correct && cmp(exp_hy, act_hy_cuda);
	}

	std::cout << "TestCase [external_test_gemma2_model_decoder_15 - CUDA]: " << (correct ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_model_decoder_16()
{
	auto path = artifact_folder_path / "test_gemma2model_decoder_16";

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers_l2(actual, expected);
			return eq;
		};

	GemmaConfig config;

	// initiate kv cache
	GemmaKVcache kv_cache;
	kv_cache.init_cache(config, 1);

	GemmaDecoderweights<float32> gd_weights;
	gd_weights.load_weights(
		(path / "gemma2model_decoder_16.gemma2decoderlayer.input_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.pre_feedforward_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.post_feedforward_layernorm.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.post_attention_layernorm.weight.dat").string(),

		(path / "gemma2model_decoder_16.gemma2decoderlayer.self_attn.q_proj.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.self_attn.k_proj.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.self_attn.v_proj.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.self_attn.o_proj.weight.dat").string(),

		(path / "gemma2model_decoder_16.gemma2decoderlayer.mlp.gate_proj.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.mlp.up_proj.weight.dat").string(),
		(path / "gemma2model_decoder_16.gemma2decoderlayer.mlp.down_proj.weight.dat").string()
	);

	int32 layer_idx = 16;
	int32 sliding_window = (layer_idx % 2 == 0 ? config.sliding_window : -1);

	// test the cache in each state
	bool correct = true;
	for (int ix = 3; ix < 31; ++ix)
	{
		std::string fn = "io_ckpt_" + std::to_string(ix);
		auto ckp_path = path / fn;

		// read tensors from files
		// read tensors from files
		auto h_hidden_states = load_tensor((ckp_path / "in_0.dat").string());
		auto h_attn_mask = load_tensor((ckp_path / "in_attention_mask.dat").string());
		auto h_pos_ids = load_tensor<int32>((ckp_path / "in_position_ids.dat").string());
		auto h_cache_pos = load_tensor<int32>((ckp_path / "in_cache_position.dat").string());

		auto exp_hy = load_tensor((ckp_path / "out_0.dat").string());

		auto d_hidden_states = h_hidden_states.copy_to_cuda();
		auto d_attn_mask = h_attn_mask.copy_to_cuda();
		auto d_pos_ids = h_pos_ids.copy_to_cuda();
		auto d_cache_pos = h_cache_pos.copy_to_cuda();

		auto act_dy_cuda = tensor_gemma_decoder(
			config,
			gd_weights,
			kv_cache,
			d_hidden_states,
			d_attn_mask,
			d_pos_ids,
			d_cache_pos,
			15,
			config.sliding_window
		);

		auto act_hy_cuda = act_dy_cuda.copy_to_host();

		// test cuda
		correct = correct && cmp(exp_hy, act_hy_cuda);
	}

	std::cout << "TestCase [external_test_gemma2_model_decoder_16 - CUDA]: " << (correct ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_slide_mask()
{
	std::filesystem::path path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\test_gemma2_slide_mask_f32";

	// read tensors from files
	// hw is tied to the embedding weights
	auto h_mask = load_tensor((path / "attention_mask.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());

	auto d_mask = h_mask.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_slide_mask(d_mask, 34);
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
	std::cout << "TestCase [external_test_gemma2_slide_mask - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_causallm()
{
	std::filesystem::path path = "/home/ubuntu/gemma2_inspected/gemma2model_whole"; //artifact_folder_path / "test_gemma2_causallm";

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers_l2(actual, expected, 0.05f);
			return eq;
		};

    GemmaConfig config;

	// initiate kv cache
    GemmaKVcache kv_cache;
    kv_cache.init_cache(config, 1);

	std::cout << "Loading weights ... \n";
    GemmaCausalLMweights<float32> gcml_weights;
    gcml_weights.load_weights_from_safetensors(
        (path / "weights_sf").string()
    );
	std::cout << "Loaded ... \n";

    // set the num_logits_to_keep
    int32 num_logits_to_keep = 1;

	// test the cache in each state
	bool correct = true;
	for (int ix = 3; ix < 35; ++ix)
	{
		std::string fn = "io_ckpt_" + std::to_string(ix);
		auto ckp_path = path / fn;

		// read tensors from files
		auto h_input_ids = load_tensor<int32>((ckp_path / "in_input_ids.dat").string());
		auto h_attn_mask = load_tensor((ckp_path / "in_attention_mask.dat").string());
		auto h_pos_ids = load_tensor<int32>((ckp_path / "in_position_ids.dat").string());
		auto h_cache_pos = load_tensor<int32>((ckp_path / "in_cache_position.dat").string());

		auto exp_hy = load_tensor((ckp_path / "out_0.dat").string());

		auto d_input_ids = h_input_ids.copy_to_cuda();
		auto d_attn_mask = h_attn_mask.copy_to_cuda();
		auto d_pos_ids = h_pos_ids.copy_to_cuda();
		auto d_cache_pos = h_cache_pos.copy_to_cuda();

		auto act_dy_cuda = tensor_gemma_causallm(
			config,
			gcml_weights,
			kv_cache,
			d_input_ids,
			d_attn_mask,
			d_pos_ids,
			d_cache_pos,
			num_logits_to_keep
		);

		auto act_hy_cuda = act_dy_cuda.copy_to_host();

		// test cuda
		correct = correct && cmp(exp_hy, act_hy_cuda);
	}

	std::cout << "TestCase [external_test_gemma2_causallm - CUDA]: " << (correct ? "PASSED" : "FAILED") << "\n";
}


void external_test_gemma2_decoder_fused_mlp()
{
	auto path = artifact_folder_path / ".." / "test_gemma2_fused_mlp_upproj_f32_sqlen158";

	// read tensors from files
	auto hx = load_tensor((path / "x.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());

	GemmaMLPweights<float32, CUDA> mlp_weights;
	mlp_weights.load_weights(
		(path / "w_gate_proj.dat").string(),
		(path / "w_up_proj.dat").string(),
		(path / "w_down_proj.dat").string()
	);

	auto dx = hx.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_mlp(mlp_weights, dx);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers_reldiff(actual, expected, 0.01f);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_gemma2_decoder_fused_mlp - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}
