#ifndef __GEMMA_LIN_SOFTCAP__
#define __GEMMA_LIN_SOFTCAP__

#include "gemma_linsoftcap.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/* linear softcapping */

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gemma_linear_softcap(
	const Tensor<dtype, CUDA>& xt,
	const Tensor<dtype, CUDA>& wt,  // [vocab_size, hidden_size]
	const dtype final_softcapping)
{
	// check sizes
	int32 dim = xt.dim;

	Shape y_shape = xt.shape;
	y_shape[dim - 1] = wt.shape[0];

	ACASSERT(xt.dim == 3, "xt should be 3 dim");
	ACASSERT(xt.shape[dim - 1] == wt.shape[1], "hidden size should match for xt and wt");

	// access the data arrays
	Tensor<dtype, CUDA> yt(dim, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_gemma_linear_softcap_f32(xt, wt, final_softcapping, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __GEMMA_LIN_SOFTCAP__
