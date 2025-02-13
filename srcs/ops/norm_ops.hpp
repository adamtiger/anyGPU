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





/* group normalization */

/*
  Group normalization.
  The implementation assumes contigous memory.
  This is a torch inspired implementation.
  @param xt: input tensor
  @param num_groups: number of groups for the channels (num_channels % num_groups = 0)
  @param wt: per-channel affine parameter (gamma)
  @param bt: same as wt but for the bias
*/
template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> tensor_group_norm(
	const Tensor<dtype, CPU>& xt,
	const int32 num_groups,
	const Tensor<dtype, CPU>& wt,  // per-channel affine parameter (gamma)
	const Tensor<dtype, CPU>& bt,  // (beta)
	const dtype eps)
{
	// check sizes
	int32 num_channels = xt.shape[1];
	ACASSERT(num_channels % num_groups == 0, "num_channels is a multiple of num_groups");
	ACASSERT(wt.shape[0] == num_channels, "wt (gamma) has to be num_channels sized");
	ACASSERT(bt.shape[0] == num_channels, "bt (beta) has to be num_channels sized");

	// calculate group sizes
	int32 dim = xt.dim;
	int32 num_group_channels = num_channels / num_groups;
	int32 instance_size = xt.stride[1];
	int32 group_size = num_group_channels * instance_size;

	int32 nbatch = xt.shape[0];

	// output
	Tensor<dtype, CPU> yt(dim, xt.shape);

	// access the dataset
	dtype* x_data = xt.buffer();
	dtype* w_data = wt.buffer();
	dtype* b_data = bt.buffer();
	dtype* y_data = yt.buffer();

	// calculate the norms
	for (int32 b = 0; b < nbatch; ++b)
	{
		for (int32 g = 0; g < num_groups; ++g)
		{
			dtype accum_sum = (dtype)0;
			dtype accum_sum_square = (dtype)0;
			int32 group_offset = b * xt.stride[0] + g * num_group_channels * xt.stride[1];

			for (int32 ix = 0; ix < group_size; ++ix)
			{
				dtype x = x_data[group_offset + ix];
				accum_sum += x;
				accum_sum_square += x * x;
			}

			dtype mean = accum_sum / (dtype)group_size;
			dtype var = accum_sum_square / (dtype)group_size - mean * mean;

			for (int32 c = 0; c < num_group_channels; ++c)
			{
				for (int32 ix = 0; ix < instance_size; ++ix)
				{
					int32 offset = group_offset + c * xt.stride[1] + ix;
					dtype x = x_data[offset];
					dtype y = (x - mean) / sqrt(var + eps);
					y = y * w_data[c] + b_data[c];
					y_data[offset] = y;
				}
			}
		}
	}

	return yt;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_group_norm(  // TODO: implementation!
	const Tensor<dtype, CUDA>& xt,
	const int32 num_groups,
	const Tensor<dtype, CUDA>& wt,  // per-channel affine parameter (gamma)
	const Tensor<dtype, CUDA>& bt,  // (beta)
	const dtype eps)
{
	// check and modify axis if needed
	int32 dim = xt.dim;
	ACASSERT(wt.dim == 1, "weight dimension is not 1");
	ACASSERT(wt.dim == bt.dim, "weight and bias must have the same dimension");
	ACASSERT(wt.shape[0] == bt.shape[0], "weight and bias must have the same size");
	ACASSERT(xt.shape[1] == wt.shape[0], "weight must have a size equal with the number of channels");
	ACASSERT(xt.shape[1] % num_groups == 0, "the number of channels is the multiple of num_groups");

	// access the data arrays
	Tensor<dtype, CUDA> yt(dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_group_norm_f32(xt, wt, bt, num_groups, eps, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}


#endif  // __NORM_OPS__
