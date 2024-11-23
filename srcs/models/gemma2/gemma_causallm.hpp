#ifndef __GEMMA_CAUSAL_LM__
#define __GEMMA_CAUSAL_LM__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"

#include "gemma_model.hpp"
#include "gemma_linsoftcap.hpp"
#include "gemma_config.hpp"


template<FloatingPointType dtype>
struct GemmaCausalLMweights
{
	GemmaModelweights<dtype> gemma_model;

	void load_weights_from_safetensors(const std::string& weights_folder_path)
	{
		gemma_model.load_weights_from_safetensors(weights_folder_path);
	}

	void load_weights_from_datfiles(const std::string& weights_folder_path)
	{
		gemma_model.load_weights_from_datfiles(weights_folder_path);
	}

	const Tensor<dtype, CUDA>& get_lm_head_weight() const  // tied to the embedding weight
	{
		return gemma_model.embedding_data;
	}
};

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gemma_causallm(  // Gemma2ForCausalLM
	const GemmaConfig& config,
	const GemmaCausalLMweights<dtype>& model_weights,
	GemmaKVcache& kv_cache,
	const Tensor<int32, CUDA>& input_ids,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const Tensor<int32, CUDA>& cache_position,
	const int32 num_logits_to_keep)
{
	// execute model
	auto hidden_states = tensor_gemma_model(
		config,
		model_weights.gemma_model,
		kv_cache, 
		input_ids,
		attention_mask,
		position_ids,
		cache_position
	);

	// lm_head (slicing can save computation)
	int seq_len = hidden_states.shape[1];
	auto hs_sliced = tensor_slice(hidden_states, 1, seq_len - num_logits_to_keep, seq_len);
	auto yt = tensor_gemma_linear_softcap(
		hs_sliced,
		model_weights.get_lm_head_weight(), 
		config.final_softcapping);

	return yt;
}

#endif // __GEMMA_CAUSAL_LM__
