#ifndef __ZAMBA_ATTN_DECDER__
#define __ZAMBA_ATTN_DECDER__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"

#include "zamba_mlp.hpp"
#include "zamba_sdpa.hpp"

template<FloatingPointType dtype>
struct ZambaAttentionDecoderweights
{
	ZambaSDPAweights<dtype> sdpa_weights;
	ZambaMLPweights<dtype, CUDA> mlp_weights;
	Tensor<dtype, CUDA> input_rmsnorm_weight;
	Tensor<dtype, CUDA> preff_rmsnorm_weight;

	void load_weights(
		const std::string& ad_path_in_rmsnorm_weight,
		const std::string& ad_path_preff_rmsnorm_weight,
		const std::string& sdpa_path_q_proj_weight,
		const std::string& sdpa_path_k_proj_weight,
		const std::string& sdpa_path_v_proj_weight,
		const std::string& sdpa_path_o_proj_weight,
		const std::vector<std::string>& sdpa_paths_q_lora_A_weights,
		const std::vector<std::string>& sdpa_paths_q_lora_B_weights,
		const std::vector<std::string>& sdpa_paths_k_lora_A_weights,
		const std::vector<std::string>& sdpa_paths_k_lora_B_weights,
		const std::vector<std::string>& sdpa_paths_v_lora_A_weights,
		const std::vector<std::string>& sdpa_paths_v_lora_B_weights,
		const std::string& mlp_path_fc1_weight,
		const std::string& mlp_path_fc2_weight,
		const std::vector<std::string>& mlp_paths_fc1_lora_A_weights,
		const std::vector<std::string>& mlp_paths_fc1_lora_B_weights)
	{
		sdpa_weights.load_weights(
			sdpa_path_q_proj_weight,
			sdpa_path_k_proj_weight,
			sdpa_path_v_proj_weight,
			sdpa_path_o_proj_weight,
			sdpa_paths_q_lora_A_weights,
			sdpa_paths_q_lora_B_weights,
			sdpa_paths_k_lora_A_weights,
			sdpa_paths_k_lora_B_weights,
			sdpa_paths_v_lora_A_weights,
			sdpa_paths_v_lora_B_weights
		);

		mlp_weights.load_weights(
			mlp_path_fc1_weight,
			mlp_path_fc2_weight,
			mlp_paths_fc1_lora_A_weights,
			mlp_paths_fc1_lora_B_weights
		);

		auto load_w = [&](const std::string& path_w)
			{
				return load_tensor<dtype>(path_w).copy_to_device<CUDA>();
			};

		input_rmsnorm_weight = load_w(ad_path_in_rmsnorm_weight);
		preff_rmsnorm_weight = load_w(ad_path_preff_rmsnorm_weight);
	}
};

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_zamba_attention_decoder(  // Zamba2AttentionDecoderLayer
	const ZambaAttentionDecoderweights<dtype>& ad_weights,
	const ZambaKVcache& kv_cache,
	const Tensor<dtype, CUDA>& hidden_states,
	const Tensor<dtype, CUDA>& orig_hidden_states,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const int hdim,
	const int fwd_layer_idx,
	const int rope_base,
	const dtype rms_norm_eps)
{
	auto hidden_states_cat = tensor_concat(hidden_states, orig_hidden_states);


	// rms norm + attention

	auto hidden_states_rmsn = tensor_rms_norm(
		hidden_states_cat, 
		-1, 
		ad_weights.input_rmsnorm_weight, 
		rms_norm_eps);
	
	auto hidden_states_atn = tensor_zamba_sdpa(
		ad_weights.sdpa_weights, 
		kv_cache, 
		hidden_states_rmsn, 
		attention_mask, 
		position_ids, 
		hdim, 
		fwd_layer_idx, 
		rope_base);


	// rms norm + mlp

	auto hidden_states_atn_rmsn = tensor_rms_norm(
		hidden_states_atn,
		-1,
		ad_weights.preff_rmsnorm_weight,
		rms_norm_eps);

	auto y = tensor_zamba_mlp(
		ad_weights.mlp_weights,
		hidden_states_atn_rmsn,
		fwd_layer_idx
	);

	return y;
}

#endif // __ZAMBA_ATTN_DECDER__
