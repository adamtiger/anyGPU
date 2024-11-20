#ifndef __GEMMA_UPDATE_MASK__
#define __GEMMA_UPDATE_MASK__

#include "gemma_update_mask.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/* update causal mask */

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gemma_update_mask(
	const Tensor<dtype, CUDA>& attention_mask,  // [batch, seq_len(?)]
	const Tensor<dtype, CUDA>& input_tensor,    // [batch, seq_len, hidden_size]
	const Tensor<int32, CUDA>& cache_position,  // [seq_len]
	const int32 target_length)
{
	// check sizes
	int batch = input_tensor.shape[0];
	int seq_len = input_tensor.shape[1];

	ACASSERT(attention_mask.shape[0] == batch, "attention's batch size is unexpected");
	ACASSERT(cache_position.shape[0] == seq_len, "cache position size is unexpected");

	// set output shape
	int y_dim = 4;
	Shape y_shape = { 
		batch,
		1, 
		seq_len,
		target_length 
	};

	// access the data arrays
	Tensor<dtype, CUDA> yt(y_dim, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_gemma_update_mask_f32(
			attention_mask, 
			cache_position, 
			target_length, 
			yt
		);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __GEMMA_UPDATE_MASK__
