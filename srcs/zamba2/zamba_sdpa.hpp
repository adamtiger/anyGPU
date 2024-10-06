#ifndef __ZAMBA_SDPA__
#define __ZAMBA_SDPA__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "transp_ops.hpp"
#include "binary_ops.hpp"
#include "mm_ops.hpp"
#include "sdp.hpp"
#include "zamba_rotary.hpp"

struct ZambaKVcache {};  // TODO: introduce later, when more information is available


template<FloatingPointType dtype>
struct ZambaSDPAweights
{
	Tensor<dtype, CUDA> q_proj_weight;
	Tensor<dtype, CUDA> k_proj_weight;
	Tensor<dtype, CUDA> v_proj_weight;
	Tensor<dtype, CUDA> o_proj_weight;
	std::vector<Tensor<dtype, CUDA>> q_lora_A_weights;
	std::vector<Tensor<dtype, CUDA>> q_lora_B_weights;
	std::vector<Tensor<dtype, CUDA>> k_lora_A_weights;
	std::vector<Tensor<dtype, CUDA>> k_lora_B_weights;
	std::vector<Tensor<dtype, CUDA>> v_lora_A_weights;
	std::vector<Tensor<dtype, CUDA>> v_lora_B_weights;

	void load_weights(
		const std::string& path_q_proj_weight,
		const std::string& path_k_proj_weight,
		const std::string& path_v_proj_weight,
		const std::string& path_o_proj_weight,
		const std::vector<std::string>& paths_q_lora_A_weights,
		const std::vector<std::string>& paths_q_lora_B_weights,
		const std::vector<std::string>& paths_k_lora_A_weights,
		const std::vector<std::string>& paths_k_lora_B_weights, 
		const std::vector<std::string>& paths_v_lora_A_weights,
		const std::vector<std::string>& paths_v_lora_B_weights)
	{
		auto load_w = [&](const std::string& path_w)
			{
				return tensor_transp(load_tensor<dtype>(path_w).copy_to_device<CUDA>());
			};

		q_proj_weight = load_w(path_q_proj_weight);
		k_proj_weight = load_w(path_k_proj_weight);
		v_proj_weight = load_w(path_v_proj_weight);
		o_proj_weight = load_w(path_o_proj_weight);

		ACASSERT(
			paths_q_lora_A_weights.size() == paths_q_lora_B_weights.size(),
			"loraA and loraB for q needs to have the same number of blocks"
		);

		ACASSERT(
			paths_k_lora_A_weights.size() == paths_k_lora_B_weights.size(),
			"loraA and loraB for k needs to have the same number of blocks"
		);

		ACASSERT(
			paths_v_lora_A_weights.size() == paths_v_lora_B_weights.size(),
			"loraA and loraB for v needs to have the same number of blocks"
		);

		size_t nw = paths_q_lora_A_weights.size();
		q_lora_A_weights.reserve(nw);
		q_lora_B_weights.reserve(nw);
		k_lora_A_weights.reserve(nw);
		k_lora_B_weights.reserve(nw);
		v_lora_A_weights.reserve(nw);
		v_lora_B_weights.reserve(nw);
		for (size_t ix = 0; ix < nw; ++ix)
		{
			q_lora_A_weights.push_back(load_w(paths_q_lora_A_weights[ix]));
			q_lora_B_weights.push_back(load_w(paths_q_lora_B_weights[ix]));

			k_lora_A_weights.push_back(load_w(paths_k_lora_A_weights[ix]));
			k_lora_B_weights.push_back(load_w(paths_k_lora_B_weights[ix]));

			v_lora_A_weights.push_back(load_w(paths_v_lora_A_weights[ix]));
			v_lora_B_weights.push_back(load_w(paths_v_lora_B_weights[ix]));
		}
	}
};

template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_zamba_sdpa(
	const ZambaSDPAweights<dtype>& sdpa_weights,
	const ZambaKVcache& kv_cache,
	const Tensor<dtype, CUDA>& hidden_states,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const int hdim,
	const int fwd_layer_idx,
	const int rope_base)
{
	// projection of hidden states

	auto q_lora_a = tensor_linear(hidden_states, sdpa_weights.q_lora_A_weights[fwd_layer_idx]);
	auto q_lora = tensor_linear(q_lora_a, sdpa_weights.q_lora_B_weights[fwd_layer_idx]);
	auto q_proj = tensor_linear(hidden_states, sdpa_weights.q_proj_weight);
	auto q_proj_lora = tensor_add(q_proj, q_lora);

	auto k_lora_a = tensor_linear(hidden_states, sdpa_weights.k_lora_A_weights[fwd_layer_idx]);
	auto k_lora = tensor_linear(k_lora_a, sdpa_weights.k_lora_B_weights[fwd_layer_idx]);
	auto k_proj = tensor_linear(hidden_states, sdpa_weights.k_proj_weight);
	auto k_proj_lora = tensor_add(k_proj, k_lora);

	auto v_lora_a = tensor_linear(hidden_states, sdpa_weights.v_lora_A_weights[fwd_layer_idx]);
	auto v_lora = tensor_linear(v_lora_a, sdpa_weights.v_lora_B_weights[fwd_layer_idx]);
	auto v_proj = tensor_linear(hidden_states, sdpa_weights.v_proj_weight);
	auto v_proj_lora = tensor_add(v_proj, v_lora);


	// change shape of q, k, v for attention module

	int bsz = hidden_states.shape[0];
	int q_len = hidden_states.shape[1];
	int hsize_2 = hidden_states.shape[2];  // 2 * hidden_size
	int num_heads = sdpa_weights.q_proj_weight.shape[0] / hdim;
	int num_key_value_heads = sdpa_weights.k_proj_weight.shape[0] / hdim;

	auto q_proj_lora_view = tensor_view(q_proj_lora, { bsz, q_len, num_heads, hdim });
	auto k_proj_lora_view = tensor_view(k_proj_lora, { bsz, q_len, num_key_value_heads, hdim });
	auto v_proj_lora_view = tensor_view(v_proj_lora, { bsz, q_len, num_key_value_heads, hdim });

	auto q_plv_t = tensor_transp(q_proj_lora_view, 1, 2);
	auto k_plv_t = tensor_transp(k_proj_lora_view, 1, 2);
	auto v_plv_t = tensor_transp(v_proj_lora_view, 1, 2);


	// rotary embedding
	auto freq = tensor_zamba_precomp_rotary_embedding<dtype>(position_ids, hdim, rope_base);
	auto query_states = tensor_apply_zamba_rotary_embedding(q_plv_t, freq);
	auto key_states = tensor_apply_zamba_rotary_embedding(k_plv_t, freq);

	auto& value_states = v_plv_t;


	// scaled dot product attention

	dtype sfmx_scale = static_cast<dtype>(1.f / sqrtf(static_cast<float32>(query_states.shape[query_states.dim - 1]) / 2.f));
	
	auto attn_out = sdp_attention_masked_scaled_fwd(
		query_states,
		key_states,
		value_states,
		attention_mask,
		sfmx_scale
	);


	// output projection

	auto attn_out_t = tensor_transp(attn_out, 1, 2);
	auto attn_out_tv = tensor_view(attn_out_t, { bsz, q_len, hsize_2 });
	auto y = tensor_linear(attn_out_tv, sdpa_weights.o_proj_weight);

	return y;
}

#endif  // __ZAMBA_SDPA__
