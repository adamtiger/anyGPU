#ifndef __GEMMA_SDPA__
#define __GEMMA_SDPA__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "transp_ops.hpp"
#include "binary_ops.hpp"
#include "mm_ops.hpp"
#include "sdp.hpp"
#include "zamba_rotary.hpp"

struct GemmaKVcache {};  // TODO: introduce later, when more information is available


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
				return tensor_transp(load_tensor<dtype>(path_w).copy_to_device<CUDA>());
			};

		q_proj_weight = load_w(path_q_proj_weight);
		k_proj_weight = load_w(path_k_proj_weight);
		v_proj_weight = load_w(path_v_proj_weight);
		o_proj_weight = load_w(path_o_proj_weight);
	}
};

template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_gemma_sdpa(  // Zamba2SdpaAttention
	const GemmaSDPAweights<dtype>& sdpa_weights,
	const GemmaKVcache& kv_cache,
	const Tensor<dtype, CUDA>& hidden_states,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const int hdim,
	const int rope_base)
{
	// projection of hidden states

	auto q_proj = tensor_linear(hidden_states, sdpa_weights.q_proj_weight);
	auto k_proj = tensor_linear(hidden_states, sdpa_weights.k_proj_weight);
	auto v_proj = tensor_linear(hidden_states, sdpa_weights.v_proj_weight);


	// change shape of q, k, v for attention module

	int bsz = hidden_states.shape[0];
	int q_len = hidden_states.shape[1];
	int hsize_2 = hidden_states.shape[2];  // 2 * hidden_size
	int num_heads = sdpa_weights.q_proj_weight.shape[0] / hdim;
	int num_key_value_heads = sdpa_weights.k_proj_weight.shape[0] / hdim;

	auto q_proj_view = tensor_view(q_proj, { bsz, q_len, num_heads, hdim });
	auto k_proj_view = tensor_view(k_proj, { bsz, q_len, num_key_value_heads, hdim });
	auto v_proj_view = tensor_view(v_proj, { bsz, q_len, num_key_value_heads, hdim });

	auto q_plv_t = tensor_transp(q_proj_view, 1, 2);
	auto k_plv_t = tensor_transp(k_proj_view, 1, 2);
	auto v_plv_t = tensor_transp(v_proj_view, 1, 2);


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

#endif  // __GEMMA_SDPA__
