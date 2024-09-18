#ifndef __NORM_OPS__
#define __NORM_OPS__

#include "norm_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"
#include <limits>


/*
  Layer normalization.
  The implementation assumes contigous memory.
  This is a torch and onnx inspired implementation.
  @param xt: input tensor
  @param axis: the first axis to be normalized
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_layer_norm(
	const Tensor<dtype, CPU>& xt, 
	const int32 axis, 
	const dtype gamma, 
	const dtype beta, 
	const dtype eps)
{
	// check and modify axis if needed
	int32 dim = xt.dim;
	ACASSERT((-dim <= axis && axis < dim), "axis is out of range");
	int32 paxis = (dim + axis) % axis;
	
	// access the data arrays
	Tensor<dtype, CPU> yt(dim, xt.shape);
	dtype* y_data = yt.buffer();
	dtype* x_data = xt.buffer();

	// reference implementation
	// reliable (but slow)
	
	// indexing parameters
	int32 norm_region_size = 1;
	for (int32 k = paxis; k < dim; ++k)
	{
		norm_region_size += xt.shape[k];
	}
	int32 num_norm_regions = yt.size() / norm_region_size;

	// iterate over the regions
	for (int32 r = 0; r < num_norm_regions; ++r)
	{
		int32 region_mem_offs = r * norm_region_size;

		// calculate the mean and variance
		dtype accum_sum = (dtype)(0);
		dtype accum_sum_square = (dtype)(0);

		for (int32 ix = 0; ix < norm_region_size; ++ix)
		{
			dtype x = x_data[region_mem_offs + ix];
			accum_sum += x;
			accum_sum_square += x * x;
		}

		dtype mean = accum_sum / static_cast<dtype>(norm_region_size);
		dtype var = accum_sum_square / static_cast<dtype>(norm_region_size) - mean * mean;

		// calculate normalized values per element
		for (int32 ix = 0; ix < norm_region_size; ++ix)
		{
			dtype x = x_data[region_mem_offs + ix];
			dtype y = ((x - mean) / sqrt(var + eps)) * gamma + beta;
			y_data[region_mem_offs + ix] = y;
		}
	}

	return yt;
}


template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_layer_norm(
	const Tensor<dtype, CUDA>& xt,
	const int32 axis,
	const dtype gamma,
	const dtype beta,
	const dtype eps)
{
	// check and modify axis if needed
	int32 dim = xt.dim;
	ACASSERT((-dim <= axis && axis < dim), "axis is out of range");
	int32 paxis = (dim + axis) % axis;

	// access the data arrays
	Tensor<dtype, CPU> yt(dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_layer_norm_f32(xt, paxis, gamma, beta, eps, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return y;
}



#endif  // __NORM_OPS__
