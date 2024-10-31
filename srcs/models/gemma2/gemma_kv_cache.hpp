#ifndef __GEMMA_KV_CACHE__
#define __GEMMA_KV_CACHE__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "gemma_config.hpp"

/*
*  Represents a hybrid cache.
*  Hybrid cache has two type of update mechanism:
*    - static update
*    - sliding update
*  Gemma2 uses both update mechanism alternately.
*/
struct GemmaKVcache
{
	// cached tensors
	std::vector<Tensor<float32, CUDA>> key_cache;
	std::vector<Tensor<float32, CUDA>> value_cache;

	int32 max_cache_len;
	int32 max_batch_len;

	// initializer
	void init_cache(
		const GemmaConfig& config,
		const int32 max_batch_len,
		const int32 max_cache_len)
	{
		this->max_batch_len = max_batch_len;
		this->max_cache_len = max_cache_len;

		int32 head_dim = config.head_dim;
		if (config.head_dim < 0)  // not defined
		{
			head_dim = config.hidden_size / config.num_attention_heads;
		}

		int32 num_key_value_heads = config.num_key_value_heads;
		if (config.num_key_value_heads < 0)  // not defined
		{
			num_key_value_heads = config.num_attention_heads;
		}


		Shape global_cache_shape = { 
			max_batch_len, 
			num_key_value_heads, 
			max_cache_len, 
			head_dim 
		};

		Shape sliding_cache_shape = {
			max_batch_len,
			num_key_value_heads,
			std::min(config.sliding_window, max_cache_len),
			head_dim
		};


		key_cache.reserve(config.num_hidden_layers);
		value_cache.reserve(config.num_hidden_layers);

		for (int ix = 0; ix < config.num_hidden_layers; ++ix)
		{
			Shape& tensor_shape = (ix % 2 == 0 ? sliding_cache_shape : global_cache_shape);

			Tensor<float32, CUDA> next_key_cache(4, tensor_shape);
			Tensor<float32, CUDA> next_value_cache(4, tensor_shape);

			key_cache.push_back(next_key_cache);
			value_cache.push_back(next_value_cache);
		}
	}

	// update function


private:
	// sliding update function

	// static update function

};

#endif  // __GEMMA_KV_CACHE__
