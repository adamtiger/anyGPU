#ifndef __NORM_OPS__
#define __NORM_OPS__

#include "norm_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"
#include <limits>


/* layer normalization */

/*
  Layer normalization.
  The implementation assumes contigous memory.
  This is a torch and onnx inspired implementation.
  @param xt: input tensor
  @param axis: the first axis to be normalized
*/
template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> tensor_layer_norm(
	const Tensor<dtype, CPU>& xt, 
	const int32 axis, 
	const Tensor<dtype, CPU> wt,
	const Tensor<dtype, CPU> bt,
	const dtype eps)
{
	// check and modify axis if needed
	int32 dim = xt.dim;
	ACASSERT((-dim <= axis && axis < dim), "axis is out of range");
	int32 paxis = (dim + axis) % dim;
	
	// access the data arrays
	Tensor<dtype, CPU> yt(dim, xt.shape);
	dtype* y_data = yt.buffer();
	dtype* x_data = xt.buffer();
	dtype* w_data = wt.buffer();
	dtype* b_data = bt.buffer();

	// reference implementation
	// reliable (but slow)
	
	// indexing parameters
	int32 norm_region_size = 1;
	for (int32 k = paxis; k < dim; ++k)
	{
		norm_region_size *= xt.shape[k];
		ACASSERT(xt.shape[k] == wt.shape[k - paxis], "wt shape is incorrect");
		ACASSERT(xt.shape[k] == bt.shape[k - paxis], "bt shape is incorrect");
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
			dtype w = w_data[ix];
			dtype b = b_data[ix];
			dtype y = ((x - mean) / sqrt(var + eps)) * w + b;
			y_data[region_mem_offs + ix] = y;
		}
	}

	return yt;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_layer_norm(
	const Tensor<dtype, CUDA>& xt,
	const int32 axis,
	const Tensor<dtype, CUDA> wt,
	const Tensor<dtype, CUDA> bt,
	const dtype eps)
{
	// check and modify axis if needed
	int32 dim = xt.dim;
	ACASSERT((-dim <= axis && axis < dim), "axis is out of range");
	int32 paxis = (dim + axis) % dim;

	for (int32 k = paxis; k < dim; ++k)
	{
		ACASSERT(xt.shape[k] == wt.shape[k - paxis], "wt shape is incorrect");
		ACASSERT(xt.shape[k] == bt.shape[k - paxis], "bt shape is incorrect");
	}

	// access the data arrays
	Tensor<dtype, CUDA> yt(dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_layer_norm_f32(xt, wt, bt, eps, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}





/* rms normalization */

/*
  RMS normalization.
  The implementation assumes contigous memory.
  This is a torch and transformer_engine inspired implementation.
  @param xt: input tensor
  @param axis: the first axis to be normalized
*/
template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> tensor_rms_norm(
	const Tensor<dtype, CPU>& xt,
	const int32 axis,
	const Tensor<dtype, CPU> wt,
	const dtype eps,
	const bool zero_centered=false)
{
	// check and modify axis if needed
	int32 dim = xt.dim;
	ACASSERT((-dim <= axis && axis < dim), "axis is out of range");
	int32 paxis = (dim + axis) % dim;

	// access the data arrays
	Tensor<dtype, CPU> yt(dim, xt.shape);
	dtype* y_data = yt.buffer();
	dtype* x_data = xt.buffer();
	dtype* w_data = wt.buffer();

	dtype delta = (zero_centered ? (dtype)1.0 : (dtype)0.0);

	// reference implementation
	// reliable (but slow)

	// indexing parameters
	int32 norm_region_size = 1;
	for (int32 k = paxis; k < dim; ++k)
	{
		norm_region_size *= xt.shape[k];
		ACASSERT(xt.shape[k] == wt.shape[k - paxis], "wt shape is incorrect");
	}
	int32 num_norm_regions = yt.size() / norm_region_size;

	// iterate over the regions
	for (int32 r = 0; r < num_norm_regions; ++r)
	{
		int32 region_mem_offs = r * norm_region_size;

		// calculate the mean and variance
		dtype accum_sum_square = (dtype)(0);

		for (int32 ix = 0; ix < norm_region_size; ++ix)
		{
			dtype x = x_data[region_mem_offs + ix];
			accum_sum_square += x * x;
		}

		dtype mean_sum_square = accum_sum_square / static_cast<dtype>(norm_region_size);

		// calculate normalized values per element
		for (int32 ix = 0; ix < norm_region_size; ++ix)
		{
			dtype x = x_data[region_mem_offs + ix];
			dtype w = w_data[ix] + delta;
			dtype y = (x / sqrt(mean_sum_square + eps)) * w;
			y_data[region_mem_offs + ix] = y;
		}
	}

	return yt;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_rms_norm(
	const Tensor<dtype, CUDA>& xt,
	const int32 axis,
	const Tensor<dtype, CUDA> wt,
	const dtype eps,
	const bool zero_centered = false)
{
	// check and modify axis if needed
	int32 dim = xt.dim;
	ACASSERT((-dim <= axis && axis < dim), "axis is out of range");
	int32 paxis = (dim + axis) % dim;

	for (int32 k = paxis; k < dim; ++k)
	{
		ACASSERT(xt.shape[k] == wt.shape[k - paxis], "wt shape is incorrect");
	}

	// access the data arrays
	Tensor<dtype, CUDA> yt(dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_rms_norm_f32(xt, wt, eps, zero_centered, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}



#endif  // __NORM_OPS__
