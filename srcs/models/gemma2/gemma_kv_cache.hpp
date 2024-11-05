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
		const int32 max_batch_len)
	{
		this->max_batch_len = max_batch_len;
		max_cache_len = config.target_length;

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
	void update_cache(
		const Tensor<float32, CUDA>& key_states,
		const Tensor<float32, CUDA>& value_states,
		const Tensor<int32, CUDA>& cache_position,
		const int32 layer_idx,
		const int32 sliding_window,
		Tensor<float32, CUDA>& k_out,
		Tensor<float32, CUDA>& v_out)
	{
		auto cache_position_cpu = cache_position.copy_to_host();

		if (sliding_window > -1)
		{
			update_cache_sliding(
				key_states,
				value_states,
				cache_position_cpu,
				layer_idx,
				k_out,
				v_out
			);
		}
		else
		{
			update_cache_static(
				key_states,
				value_states,
				cache_position_cpu,
				layer_idx,
				k_out,
				v_out
			);
		}
	}

private:
	// sliding update function
	void update_cache_sliding(
		const Tensor<float32, CUDA>& key_states,
		const Tensor<float32, CUDA>& value_states,
		const Tensor<int32, CPU>& cache_position,
		const int32 layer_idx,
		Tensor<float32, CUDA>& k_out,
		Tensor<float32, CUDA>& v_out)
	{
		auto& k = key_cache.at(layer_idx);
		auto& v = value_cache.at(layer_idx);

		const int32 state_len = cache_position.numel();

		if (state_len > max_cache_len)
		{
			std::vector<int32> srcs(max_cache_len);
			for (int ix = 0; ix < max_cache_len; ++ix)
			{
				srcs[ix] = state_len - max_cache_len + ix;
			}
			Tensor<int32, CPU> src_indices({ (int64)max_cache_len }, srcs);

			auto trg_indices = create_default_indices(max_cache_len);

			copy_data_to_cache(key_states, src_indices, trg_indices, k);
			copy_data_to_cache(value_states, src_indices, trg_indices, v);
		}


		Tensor<float32, CUDA> k_slided(k.dim, k.shape, k.stride);
		Tensor<float32, CUDA> v_slided(v.dim, v.shape, v.stride);
		{
			Tensor<int32, CPU> src_indices = create_default_indices(max_cache_len);
			Tensor<int32, CPU> trg_indices = create_default_indices(max_cache_len);

			// shift for sliding if needed
			if (cache_position.buffer()[state_len - 1] >= max_cache_len - 1)
			{
				for (int ix = 0; ix < max_cache_len; ++ix)
				{
					int32& src = src_indices.buffer()[ix];
					src = (src + 1) % max_cache_len;
				}
			}

			copy_data_to_cache(k, src_indices, trg_indices, k_slided);
			copy_data_to_cache(v, src_indices, trg_indices, v_slided);
		}

		
		{
			Tensor<int32, CPU> src_indices = create_default_indices(state_len);
			Tensor<int32, CPU> trg_indices = create_default_indices(state_len);

			for (int ix = 0; ix < state_len; ++ix)
			{
				int32 cp = cache_position.buffer()[ix];
				int32& trg = trg_indices.buffer()[ix];
				
				trg = std::min(std::max(cp, 0), max_cache_len - 1);  // clamping
			}

			copy_data_to_cache(key_states, src_indices, trg_indices, k_slided);
			copy_data_to_cache(value_states, src_indices, trg_indices, v_slided);
		}


		k = k_slided;  // store in cache
		v = v_slided;

		k_out = k;
		v_out = v;
	}

	// static update function
	void update_cache_static(
		const Tensor<float32, CUDA>& key_states,
		const Tensor<float32, CUDA>& value_states,
		const Tensor<int32, CPU>& cache_position,
		const int32 layer_idx,
		Tensor<float32, CUDA>& k_out,
		Tensor<float32, CUDA>& v_out)
	{
		auto& k = key_cache.at(layer_idx);
		auto& v = value_cache.at(layer_idx);

		auto src_indices = create_default_indices(cache_position.numel());
		copy_data_to_cache(key_states, src_indices, cache_position, k);
		copy_data_to_cache(value_states, src_indices, cache_position, v);

		k_out = k;
		v_out = v;
	}

	// copy data according to a mapping index
	// cache[:, :, trg_indices, :] = states[:, :, src_indices, :]
	void copy_data_to_cache(
		const Tensor<float32, CUDA>& states,
		const Tensor<int32, CPU>& src_indices,
		const Tensor<int32, CPU>& trg_indices,
		Tensor<float32, CUDA>& cache)
	{
		ACASSERT(src_indices.numel() == trg_indices.numel(), "src and trg indices needs to have the same size");

		int32 L = states.shape[0] * states.shape[1];  // cache should give the same
		int32 N = src_indices.numel();
		int32 H = states.shape[3];  // number of elements to copy in a step

		int32 src_stride_L = states.stride[1];
		int32 src_stride_N = states.stride[2];

		int32 trg_stride_L = cache.stride[1];
		int32 trg_stride_N = cache.stride[2];

		float32* src_data = states.buffer();
		float32* trg_data = cache.buffer();

		for (int32 i = 0; i < L; ++i)
		{
			for (int32 j = 0; j < N; ++j)
			{
				int32 trg_j = trg_indices.buffer()[j];
				int32 src_j = src_indices.buffer()[j];

				cudaMemcpy(
					trg_data + i * trg_stride_L + trg_j * trg_stride_N,
					src_data + i * src_stride_L + src_j * src_stride_N,
					sizeof(float32) * H,
					cudaMemcpyDeviceToDevice
				);
			}
		}
	}

	Tensor<int32, CPU> create_default_indices(const int32 length)
	{
		std::vector<int> hdata(length);
		for (int ix = 0; ix < length; ++ix)
		{
			hdata[ix] = ix;
		}

		Tensor<int32, CPU> indices({ (int64)length }, hdata);
		return indices;
	}
};

#endif  // __GEMMA_KV_CACHE__
