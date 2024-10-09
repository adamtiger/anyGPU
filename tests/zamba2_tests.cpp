#include "zamba2_tests.hpp"
#include "test_tools.hpp"
#include "dat_file.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <filesystem>

#include "zamba_glu.hpp"
#include "zamba_mlp.hpp"
#include "zamba_rotary.hpp"
#include "zamba_sdpa.hpp"
#include "zamba_attn_decoder.hpp"
#include "zamba_gated_rmsnorm.hpp"
#include "causal_conv1d.hpp"
#include "sdp.hpp"


const std::filesystem::path artifact_folder_path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\zamba2_tests";

void external_test_zamba2_model_rmsnorm()
{
	auto path = artifact_folder_path / "test_zamba2model_rmsnorm";

	// read tensors from files
	auto hw = load_tensor((path / "zamba2model_rmsnorm.zamba2rmsnorm.weight.dat").string());
	auto hx = load_tensor((path / "in_0.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_rms_norm<float32>(dx, -1, dw, 1e-5f);
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
	std::cout << "TestCase [external_test_zamba2_model_rmsnorm - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_zamba2_attndeco_attn()
{
	auto path = artifact_folder_path / "test_zamba2AttenDecodLy_attn";

	// read tensors from files
	auto h_hidden_states = load_tensor((path / "in_0.dat").string());
	auto h_attn_mask = load_tensor((path / "in_attention_mask.dat").string());
	auto h_pos_ids = load_tensor<int32>((path / "in_position_ids.dat").string());

	auto exp_hy = load_tensor((path / "out_0.dat").string());

	ZambaKVcache kv_cache = {};

	ZambaSDPAweights<float32> sdpa_weights;
	sdpa_weights.load_weights(
		(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.q_proj.weight.dat").string(),
		(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.k_proj.weight.dat").string(),
		(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.v_proj.weight.dat").string(),
		(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.o_proj.weight.dat").string(),
		{
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_A_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_A_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_A_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_A_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_A_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_B_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_B_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_B_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_B_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_B_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_q_lora_B_list.5.weight.dat").string()
		},
		{
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_A_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_A_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_A_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_A_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_A_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_B_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_B_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_B_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_B_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_B_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_k_lora_B_list.5.weight.dat").string()
		},
		{
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_A_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_A_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_A_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_A_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_A_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_B_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_B_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_B_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_B_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_B_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_attn.zamba2sdpaattention.linear_v_lora_B_list.5.weight.dat").string()
		}
	);

	const int hdim = 128;
	const int fwd_layer_idx = 0;
	const int rope_base = 10000;

	auto d_hidden_states = h_hidden_states.copy_to_cuda();
	auto d_attn_mask = h_attn_mask.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();

	auto act_dy_cuda = tensor_zamba_sdpa(
		sdpa_weights, 
		kv_cache, 
		d_hidden_states, 
		d_attn_mask, 
		d_pos_ids, 
		hdim, 
		fwd_layer_idx, 
		rope_base
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
	std::cout << "TestCase [external_test_zamba2_attndeco_attn - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_zamba2_attndeco_mlp()
{
	auto path = artifact_folder_path / "test_zamba2AttenDecodLy_mlp";

	// read tensors from files
	auto hx = load_tensor((path / "in_0.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	ZambaMLPweights<float32, CUDA> mlp_weights;
	mlp_weights.load_weights(
		(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1.weight.dat").string(),
		(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc2.weight.dat").string(),
		{
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.5.weight.dat").string()
		}
	);

	const int fwd_layer_idx = 0;

	auto dx = hx.copy_to_cuda();
	auto act_dy_cuda = tensor_zamba_mlp(mlp_weights, dx, fwd_layer_idx);
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
	std::cout << "TestCase [external_test_zamba2_attndeco_mlp - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_zamba2_attn_rotary()
{
	auto path = artifact_folder_path / "test_zamba2attention_rotary";

	// read tensors from files
	auto hp = load_tensor<int32>((path / "position_ids.dat").string());

	auto hq = load_tensor((path / "query_states_in.dat").string());
	auto exp_hq = load_tensor((path / "query_states_out.dat").string());
	auto hk = load_tensor((path / "key_states_in.dat").string());
	auto exp_hk = load_tensor((path / "key_states_out.dat").string());

	auto dp = hp.copy_to_cuda();
	auto dq = hq.copy_to_cuda();
	auto dk = hk.copy_to_cuda();

	auto freq = tensor_zamba_precomp_rotary_embedding<float32>(dp, 128);
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
	bool eq = cmp(exp_hq, act_hq_cuda);
	eq = eq && cmp(exp_hk, act_hk_cuda);
	std::cout << "TestCase [external_test_zamba2_attn_rotary - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_zamba2_attn_sdp()
{
	auto path = artifact_folder_path / "test_zamba2attention_sdp";

	// read tensors from files
	auto hq = load_tensor((path / "in_0.dat").string());
	auto hk = load_tensor((path / "in_1.dat").string());
	auto hv = load_tensor((path / "in_2.dat").string());
	auto hmask = load_tensor((path / "in_attn_mask.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto dq = hq.copy_to_cuda();
	auto dk = hk.copy_to_cuda();
	auto dv = hv.copy_to_cuda();
	auto dmask = hmask.copy_to_cuda();

	auto act_dy_cuda = sdp_attention_masked_scaled_fwd(dq, dk, dv, dmask, 0.125f);

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
	std::cout << "TestCase [external_test_zamba2_attn_sdp - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_zamba2_glu()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 2, 4, 5, 2048 }, 11);
	auto dtc = tensor_zamba_glu(dta);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htc = tensor_zamba_glu(hta);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc_from_cuda, htc);  // checks the sizes
	eq = eq && compare_data_buffers(htc_from_cuda, htc);

	std::cout << "TestCase [test_zamba2_glu]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_zamba2_model_attndecoder()
{
	auto path = artifact_folder_path / "test_zamba2model_attndecoder";

	// read tensors from files
	auto h_hidden_states = load_tensor((path / "in_0.dat").string());
	auto h_orig_hidden_states = load_tensor((path / "in_original_hidden_states.dat").string());
	auto h_attn_mask = load_tensor((path / "in_attention_mask.dat").string());
	auto h_pos_ids = load_tensor<int32>((path / "in_position_ids.dat").string());

	auto exp_hy = load_tensor((path / "out_0.dat").string());

	ZambaKVcache kv_cache = {};

	ZambaAttentionDecoderweights<float32> ad_weights;
	ad_weights.load_weights(
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.input_layernorm.weight.dat").string(),
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.pre_ff_layernorm.weight.dat").string(),
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.q_proj.weight.dat").string(),
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.k_proj.weight.dat").string(),
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.v_proj.weight.dat").string(),
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.o_proj.weight.dat").string(),
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_A_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_A_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_A_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_A_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_A_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_B_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_B_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_B_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_B_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_B_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_q_lora_B_list.5.weight.dat").string()
		},
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_A_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_A_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_A_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_A_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_A_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_B_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_B_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_B_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_B_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_B_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_k_lora_B_list.5.weight.dat").string()
		},
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_A_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_A_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_A_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_A_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_A_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_B_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_B_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_B_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_B_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_B_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.self_attn.linear_v_lora_B_list.5.weight.dat").string()
		},
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1.weight.dat").string(),
		(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc2.weight.dat").string(),
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_A_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_A_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_A_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_A_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_A_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_B_list.0.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_B_list.1.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_B_list.2.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_B_list.3.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_B_list.4.weight.dat").string(),
			(path / "zamba2model_attndecoder.zamba2attentiondecoderlayer.feed_forward.linear_fc1_lora_B_list.5.weight.dat").string()
		}
	);

	const int hdim = 128;
	const int fwd_layer_idx = 0;
	const int rope_base = 10000;
	float32 rms_norm_eps = 1e-5f;

	auto d_hidden_states = h_hidden_states.copy_to_cuda();
	auto d_orig_hidden_states = h_orig_hidden_states.copy_to_cuda();
	auto d_attn_mask = h_attn_mask.copy_to_cuda();
	auto d_pos_ids = h_pos_ids.copy_to_cuda();

	auto act_dy_cuda = tensor_zamba_attention_decoder(
		ad_weights,
		kv_cache,
		d_hidden_states,
		d_orig_hidden_states,
		d_attn_mask,
		d_pos_ids,
		hdim,
		fwd_layer_idx,
		rope_base,
		rms_norm_eps
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
	std::cout << "TestCase [test_zamba2_model_attndecoder - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_mamba2_layer_causal_conv1d()
{
	auto path = artifact_folder_path / "test_mamba2layer_causal_conv1d_fn";

	// read tensors from files
	auto hx = load_tensor((path / "in_0.dat").string());
	auto hw = load_tensor((path / "in_1.dat").string());
	auto hb = load_tensor((path / "in_bias.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto act_hy_cpu = tensor_causal_conv1d(hx, hw, hb);

	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto db = hb.copy_to_cuda();
	auto act_dy_cuda = tensor_causal_conv1d(dx, dw, db);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cpu);
	std::cout << "TestCase [external_test_mamba2_layer_causal_conv1d - CPU]: " << (eq ? "PASSED" : "FAILED") << "\n";

	eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_mamba2_layer_causal_conv1d - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_mamba2_layer_gated_rmsnorm()
{
	auto path = artifact_folder_path / "test_mamba2layer_gated_rmsnorm";

	// constant params
	const float32 eps = 1e-5f;
	const int32 gsize = 4096;  // group size

	// read tensors from files
	auto hx = load_tensor((path / "in_0.dat").string());
	auto hz = load_tensor((path / "in_1.dat").string());
	auto hw = load_tensor((path / "mamba2layer_rmsnorm.rmsnorm.weight.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto dx = hx.copy_to_cuda();
	auto dz = hz.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_zamba_gated_rmsnorm(dx, dz, dw, gsize, eps);
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
	std::cout << "TestCase [external_test_mamba2_layer_gated_rmsnorm - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}
