#ifndef __GEMMA_MODEL__
#define __GEMMA_MODEL__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"

#include "gemma_decoder.hpp"


template<FloatingPointType dtype>
struct GemmaModelweights
{
	std::vector<GemmaDecoderweights<dtype>> decoder_layers;
	Tensor<dtype, CUDA> embedding_data;
	Tensor<dtype, CUDA> pos_rmsnorm_weight;

	void load_weights(const std::string& sf_tensors_path)
	{

	}
};

template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_gemma_model(  // Gemma2Model
	const GemmaModelweights<dtype>& model_weights,
	const GemmaKVcache& kv_cache,
	const Tensor<int32, CUDA>& input_ids,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const int hdim,  // TODO: are these the same for all the decoder layers?
	const int rope_base,
	const dtype rms_norm_eps,
	const dtype sfmx_scale)
{
	// TODO: set ouput_attentions and output_hidden_states (how it is happening?)

	// embed the input ids (calculated from tokenizer)
	auto inp_embeds = tensor_embedding(input_ids, model_weights.embedding_data);


	// TODO: kv_cache creation if not exists (is it created here?)

	// TODO: cache_position creation if not exists (???)

	// TODO: position ids creation if not exists (???)

	// TODO: update_causal mask (can require implementation)


	// normalize hidden states
	dtype norm_factor = 1.0;  // TODO: calculate this! can require a config param (should config params be a separate struct?)
	auto hidden_states = tensor_mul(inp_embeds, norm_factor);

	// execute the decoder layers (one after each other)
	for (auto& decoder_weights : model_weights.decoder_layers)
	{
		auto layer_outputs = tensor_gemma_decoder(
			decoder_weights,
			kv_cache,
			hidden_states,
			attention_mask,
			position_ids,
			hdim,
			rope_base,
			rms_norm_eps,
			sfmx_scale);

		hidden_states = layer_outputs;  // TODO: may be this will be a list to return more values! (or return on the argument!)
	}

	// calculate rms norm
	auto y = tensor_rms_norm(hidden_states, -1, model_weights.pos_rmsnorm_weight, rms_norm_eps, true);


	// TODO: store required additional outputs (e.g. all hidden states, attentions etc.) (is this required???, what is the purpose? saving computation and memory?)

	return y;
}

#endif // __GEMMA_MODEL__
