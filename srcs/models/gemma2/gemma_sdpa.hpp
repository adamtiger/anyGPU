#ifndef __GEMMA_SDPA__
#define __GEMMA_SDPA__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "transp_ops.hpp"
#include "binary_ops.hpp"
#include "mm_ops.hpp"
#include "sdp.hpp"
#include "gemma_kv_cache.hpp"
#include "zamba_rotary.hpp"

#include "sdpa_gemma2_linear.cuh"


template<FloatingPointType dtype>
struct GemmaSDPAweights
{
	Tensor<dtype, CUDA> q_proj_weight;
	Tensor<dtype, CUDA> k_proj_weight;
	Tensor<dtype, CUDA> v_proj_weight;
	Tensor<dtype, CUDA> o_proj_weight;

	void load_weights(
		const std::string& path_q_proj_weight,
		const std::string& path_k_proj_weight,
		const std::string& path_v_proj_weight,
		const std::string& path_o_proj_weight)
	{
		auto load_w = [&](const std::string& path_w)
			{
				return copy_to_device<dtype, CPU, CUDA>(tensor_transp(load_tensor<dtype>(path_w)));
			};

		q_proj_weight = load_w(path_q_proj_weight);
		k_proj_weight = load_w(path_k_proj_weight);
		v_proj_weight = load_w(path_v_proj_weight);
		o_proj_weight = load_w(path_o_proj_weight);
	}

	void set_weights(
		Tensor<dtype, CUDA>& sdpa_q_proj_weight,
		Tensor<dtype, CUDA>& sdpa_k_proj_weight,
		Tensor<dtype, CUDA>& sdpa_v_proj_weight,
		Tensor<dtype, CUDA>& sdpa_o_proj_weight)
	{
		q_proj_weight = tensor_transp(sdpa_q_proj_weight);
		k_proj_weight = tensor_transp(sdpa_k_proj_weight);
		v_proj_weight = tensor_transp(sdpa_v_proj_weight);
		o_proj_weight = tensor_transp(sdpa_o_proj_weight);
	}
};


template<FloatingPointType dtype, int variant>
inline Tensor<dtype, CUDA> tensor_gemma_sdpa_linear(
	const Tensor<dtype, CUDA>& xt,
	const Tensor<dtype, CUDA>& wt)
{
	Tensor<dtype, CUDA> yt;

	if constexpr (variant == 1)
	{
		Shape y_shape = xt.shape;
		y_shape[xt.dim - 1] = wt.shape[1];
		Tensor<dtype, CUDA> out(xt.dim, y_shape);

		cu_sdpa_gemma2_linear_f32_v1(xt, wt, out);

		yt = out;
	}
	else  // default
	{
		yt = tensor_linear(xt, wt);
	}

	return yt;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gemma_sdpa(
	const GemmaConfig& config,
	const GemmaSDPAweights<dtype>& sdpa_weights,
	GemmaKVcache& kv_cache,
	const Tensor<dtype, CUDA>& hidden_states,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const Tensor<int32, CUDA>& cache_position,
	const int32 layer_idx,
	const int32 sliding_window)
{
	// projection of hidden states

	auto q_proj = tensor_gemma_sdpa_linear<dtype, 1>(hidden_states, sdpa_weights.q_proj_weight);
	auto k_proj = tensor_gemma_sdpa_linear<dtype, 1>(hidden_states, sdpa_weights.k_proj_weight);
	auto v_proj = tensor_gemma_sdpa_linear<dtype, 1>(hidden_states, sdpa_weights.v_proj_weight);


	// change shape of q, k, v for attention module

	int bsz = q_proj.shape[0];
	int q_len = q_proj.shape[1];
	int hidden_size = q_proj.shape[2];
	int num_heads = sdpa_weights.q_proj_weight.shape[1] / config.head_dim;
	int num_key_value_heads = sdpa_weights.k_proj_weight.shape[1] / config.head_dim;

	auto q_proj_view = tensor_view(q_proj, { bsz, q_len, num_heads, config.head_dim });
	auto k_proj_view = tensor_view(k_proj, { bsz, q_len, num_key_value_heads, config.head_dim });
	auto v_proj_view = tensor_view(v_proj, { bsz, q_len, num_key_value_heads, config.head_dim });

	auto q_plv_t = tensor_transp(q_proj_view, 1, 2);
	auto k_plv_t = tensor_transp(k_proj_view, 1, 2);
	auto v_plv_t = tensor_transp(v_proj_view, 1, 2);


	// rotary embedding (without the transpose, the core alt. rotary can be used)
	auto freq = tensor_zamba_precomp_rotary_embedding<dtype>(position_ids, config.head_dim, config.rope_base);
	auto query_states = tensor_apply_zamba_rotary_embedding(q_plv_t, freq);
	auto key_states_emb = tensor_apply_zamba_rotary_embedding(k_plv_t, freq);


	// updating the kv cache and getting the updated values back
	Tensor<dtype, CUDA> k_from_cache;
	Tensor<dtype, CUDA> v_from_cache;
	kv_cache.update_cache(
		key_states_emb, 
		v_plv_t, 
		cache_position, 
		layer_idx, 
		sliding_window, 
		k_from_cache, 
		v_from_cache
	);


	// repeat_kv
	auto key_states = tensor_repeat(k_from_cache, 1, 2);
	auto value_states = tensor_repeat(v_from_cache, 1, 2);

	// slicing attention mask
	auto attention_mask_sliced = tensor_slice(attention_mask, 3, 0, key_states.shape[2]);

	// scaled dot product attention
	auto attn_out = sdp_attention_masked_scaled_fwd(
		query_states,
		key_states,
		value_states,
		attention_mask_sliced,
		config.sfmx_scale
	);


	// output projection
	auto attn_out_t = tensor_transp(attn_out, 1, 2);
	auto attn_out_tv = tensor_view(attn_out_t, { bsz, q_len, hidden_size });
	auto y = tensor_gemma_sdpa_linear<dtype, 1>(attn_out_tv, sdpa_weights.o_proj_weight);

	return y;
}

#endif  // __GEMMA_SDPA__
