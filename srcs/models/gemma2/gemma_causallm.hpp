#ifndef __GEMMA_CAUSAL_LM__
#define __GEMMA_CAUSAL_LM__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"

#include "gemma_model.hpp"
#include "gemma_linsoftcap.hpp"


template<FloatingPointType dtype>
struct GemmaCausalLMweights
{
	GemmaModelweights<dtype> gemma_model;

	void load_weights(const std::string& weights_folder_path)
	{
		gemma_model.load_weights(weights_folder_path);
	}

	Tensor<dtype, CUDA>& get_lm_head_weight() const  // tied to the embedding weight
	{
		return gemma_model.embedding_data;
	}
};

template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_gemma_causallm(  // Gemma2ForCausalLM
	const GemmaCausalLMweights<dtype>& model_weights,
	const GemmaKVcache& kv_cache,
	const Tensor<int32, CUDA>& input_ids,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const int hdim,
	const int hidden_size,
	const int rope_base,
	const dtype rms_norm_eps,
	const dtype sfmx_scale, 
	const dtype final_softcapping)
{
	// execute model
	auto hidden_states = tensor_gemma_model(
		model_weights.gemma_model,
		kv_cache, 
		input_ids,
		attention_mask,
		position_ids,
		hdim,
		hidden_size,
		rope_base,
		rms_norm_eps, 
		sfmx_scale
	);

	// lm_head (slicing can save computation)
	auto yt = tensor_gemma_linear_softcap(
		hidden_states, 
		model_weights.get_lm_head_weight(), 
		final_softcapping);

	return yt;
}

#endif // __GEMMA_CAUSAL_LM__
