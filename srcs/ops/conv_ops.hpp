#ifndef __CONV_OPS__
#define __CONV_OPS__

#include "norm_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/* convolutions */

/*
  2D convolution.
*/
template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> tensor_conv2d(
	const Tensor<dtype, CPU>& xt,
	const Tensor<dtype, CPU> wt,
	const Tensor<dtype, CPU> bt,
	const std::array<int32, 2>& stride,
	const std::array<int32, 4>& pads)
{
	return {};
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_conv2d(
	const Tensor<dtype, CUDA>& xt,
	const Tensor<dtype, CUDA> wt,
	const Tensor<dtype, CUDA> bt,
	const std::array<int32, 2>& stride,
	const std::array<int32, 4>& pads)
{
	// check and modify axis if needed
	int32 dim = xt.dim;

	// access the data arrays
	Tensor<dtype, CUDA> yt(dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		//cu_tensor_layer_norm_f32(xt, wt, bt, eps, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}


#endif  // __CONV_OPS__

