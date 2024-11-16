#ifndef __GEMMA_SLIDE_MASK__
#define __GEMMA_SLIDE_MASK__

#include "gemma_slide_mask.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/* slide attention mask */

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gemma_slide_mask(
	const Tensor<dtype, CUDA>& attention_mask,  // [batch, 1, seq_len, target_length]
	const int32 sliding_window)
{
	ACASSERT(attention_mask.dim == 4, "attention has to be 4 dimensional");

	int32 seq_len = attention_mask.shape[2];

	if (seq_len <= sliding_window)
	{
		return attention_mask;
	}

	// access the data arrays
	Tensor<dtype, CUDA> yt(attention_mask.dim, attention_mask.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_gemma_slide_mask_f32(
			attention_mask,
			sliding_window,
			yt
		);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __GEMMA_SLIDE_MASK__
