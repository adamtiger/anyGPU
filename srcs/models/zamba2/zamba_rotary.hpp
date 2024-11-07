#ifndef __ZAMBA_ROTARY__
#define __ZAMBA_ROTARY__

#include "zamba_rotary.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_zamba_precomp_rotary_embedding(
	const Tensor<int32, CUDA>& pt,  // position ids, shape: [batch, seq_len]
	const int32 emb_size,
	const int32 base = 10000)
{
	// access the data arrays
	Tensor<dtype, CUDA> ft(3, { pt.shape[0], pt.shape[1], emb_size / 2 });

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_zamba_precomp_rotary_embedding_f32(pt, emb_size, base, ft);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return ft;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_apply_zamba_rotary_embedding(
	const Tensor<dtype, CUDA>& xt,  // input (e.g. query or key), shape: [batch, num_heads, seq_len, hdim]
	const Tensor<dtype, CUDA>& ft)  // [batch, seq_len, hdim / 2]
{
	// access the data arrays
	Tensor<dtype, CUDA> yt(xt.dim, xt.shape);  // yt has the same shape as the input

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_apply_zamba_rotary_embedding_f32(xt, ft, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __ZAMBA_ROTARY__
