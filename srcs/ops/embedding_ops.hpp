#ifndef __EMBEDDING_OPS__
#define __EMBEDDING_OPS__

#include "embedding_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/*
  Embedding operator.
  @param xt: input tensor with indices
  @param wt: embedding weights
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_embedding(const Tensor<int32, CPU>& xt, const Tensor<dtype, CPU>& wt)
{
	// access the data arrays
	const int length = xt.size();
	const int emb_size = wt.shape[1];

	Shape y_shape = xt.shape;
	y_shape[xt.dim] = emb_size;
	Tensor<dtype, CPU> yt(xt.dim + 1, y_shape);
	dtype* y_data = yt.buffer();
	int32* x_data = xt.buffer();
	dtype* w_data = wt.buffer();

	// reference implementation
	// reliable (but slow)

	for (int k = 0; k < length; ++k)
	{
		int32 i = x_data[k];
		memcpy(y_data + k * emb_size, w_data + i * emb_size, emb_size * sizeof(dtype));
	}

	return yt;
}


template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_embedding(const Tensor<int32, CUDA>& xt, const Tensor<dtype, CUDA>& wt)
{
	// access the data arrays
	Shape y_shape = xt.shape;
	y_shape[xt.dim] = wt.shape[1];
	Tensor<dtype, CUDA> yt(xt.dim + 1, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_embedding_f32(xt, wt, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __EMBEDDING_OPS__
