#ifndef __GEMMA_DECDER__
#define __GEMMA_DECDER__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"

#include "gemma_mlp.hpp"
#include "gemma_sdpa.hpp"
#include "gemma_config.hpp"
#include "gemma_slide_mask.hpp"


template<FloatingPointType dtype>
struct GemmaDecoderweights
{
	GemmaSDPAweights<dtype> sdpa_weights;
	GemmaMLPweights<dtype, CUDA> mlp_weights;
	Tensor<dtype, CUDA> input_rmsnorm_weight;
	Tensor<dtype, CUDA> preff_rmsnorm_weight;
	Tensor<dtype, CUDA> postf_rmsnorm_weight;
	Tensor<dtype, CUDA> post_attn_rmsnorm_weight;

	void load_weights(
		const std::string& gd_path_in_rmsnorm_weight,
		const std::string& gd_path_preff_rmsnorm_weight,
		const std::string& gd_path_postf_rmsnorm_weight,
		const std::string& gd_path_post_attn_rmsnorm_weight,
		const std::string& sdpa_path_q_proj_weight,
		const std::string& sdpa_path_k_proj_weight,
		const std::string& sdpa_path_v_proj_weight,
		const std::string& sdpa_path_o_proj_weight,
		const std::string& mlp_path_gate_proj_weight,
		const std::string& mlp_path_up_proj_weight,
		const std::string& mlp_path_down_proj_weight)
	{
		sdpa_weights.load_weights(
			sdpa_path_q_proj_weight,
			sdpa_path_k_proj_weight,
			sdpa_path_v_proj_weight,
			sdpa_path_o_proj_weight
		);

		mlp_weights.load_weights(
			mlp_path_gate_proj_weight,
			mlp_path_up_proj_weight,
			mlp_path_down_proj_weight
		);

		auto load_w = [&](const std::string& path_w)
			{
				return copy_to_device<dtype, CPU, CUDA>(load_tensor<dtype>(path_w));
			};

		input_rmsnorm_weight = load_w(gd_path_in_rmsnorm_weight);
		preff_rmsnorm_weight = load_w(gd_path_preff_rmsnorm_weight);
		postf_rmsnorm_weight = load_w(gd_path_postf_rmsnorm_weight);
		post_attn_rmsnorm_weight = load_w(gd_path_post_attn_rmsnorm_weight);
	}

	void set_weights(
		Tensor<dtype, CUDA>& gd_in_rmsnorm_weight,
		Tensor<dtype, CUDA>& gd_preff_rmsnorm_weight,
		Tensor<dtype, CUDA>& gd_postf_rmsnorm_weight,
		Tensor<dtype, CUDA>& gd_post_attn_rmsnorm_weight,
		Tensor<dtype, CUDA>& sdpa_q_proj_weight,
		Tensor<dtype, CUDA>& sdpa_k_proj_weight,
		Tensor<dtype, CUDA>& sdpa_v_proj_weight,
		Tensor<dtype, CUDA>& sdpa_o_proj_weight,
		Tensor<dtype, CUDA>& mlp_gate_proj_weight,
		Tensor<dtype, CUDA>& mlp_up_proj_weight,
		Tensor<dtype, CUDA>& mlp_down_proj_weight)
	{
		sdpa_weights.set_weights(
			sdpa_q_proj_weight,
			sdpa_k_proj_weight,
			sdpa_v_proj_weight,
			sdpa_o_proj_weight
		);

		mlp_weights.set_weights(
			mlp_gate_proj_weight,
			mlp_up_proj_weight,
			mlp_down_proj_weight
		);

		input_rmsnorm_weight = gd_in_rmsnorm_weight;
		preff_rmsnorm_weight = gd_preff_rmsnorm_weight;
		postf_rmsnorm_weight = gd_postf_rmsnorm_weight;
		post_attn_rmsnorm_weight = gd_post_attn_rmsnorm_weight;
	}

};

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gemma_decoder(  // Gemma2DecoderLayer
	const GemmaConfig& config,
	const GemmaDecoderweights<dtype>& decoder_weights,
	GemmaKVcache& kv_cache,
	const Tensor<dtype, CUDA>& hidden_states,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const Tensor<int32, CUDA>& cache_position,
	const int32 layer_idx,
	const int32 sliding_window)
{ 
	// sliding mask update (if needed)
	Tensor<dtype, CUDA> s_attention_mask = attention_mask;
	if (layer_idx % 2 == 0)
	{
		s_attention_mask = tensor_gemma_slide_mask(attention_mask, sliding_window);
	}

	// rms norm + attention + rms norm

	auto hidden_states_inp = tensor_rms_norm(
		hidden_states,
		-1,
		decoder_weights.input_rmsnorm_weight,
		config.rms_norm_eps,
		true);

	auto hidden_states_sdpa = tensor_gemma_sdpa(
		config,
		decoder_weights.sdpa_weights,
		kv_cache,
		hidden_states_inp,
		s_attention_mask,
		position_ids,
		cache_position,
		layer_idx,
		sliding_window);

	auto hidden_states_post_attn = tensor_rms_norm(
		hidden_states_sdpa,
		-1,
		decoder_weights.post_attn_rmsnorm_weight,
		config.rms_norm_eps,
		true);

	auto hidden_states_after = tensor_add(hidden_states, hidden_states_post_attn);

	// rms norm + mlp + rms norm

	auto hidden_states_preff = tensor_rms_norm(
		hidden_states_after,
		-1,
		decoder_weights.preff_rmsnorm_weight,
		config.rms_norm_eps,
		true);

	auto hidden_states_mlp = tensor_gemma_mlp(
		decoder_weights.mlp_weights,
		hidden_states_preff);

	auto hidden_states_postf = tensor_rms_norm(
		hidden_states_mlp,
		-1,
		decoder_weights.postf_rmsnorm_weight,
		config.rms_norm_eps,
		true);

	auto y = tensor_add(hidden_states_after, hidden_states_postf);

	return y;
}

#endif // __GEMMA_DECDER__
