#ifndef __MATH_OPS__
#define __MATH_OPS__

#include "math_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/*
  SiLU activation (elementwise op).
  silu(x) = x * sigmoid(x)
  sigmoid(x) = 1 / (1 + exp(-x))
  @param xt: input tensor
*/
template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> tensor_silu(const Tensor<dtype, CPU>& xt)
{
	// access the data arrays
	Tensor<dtype, CPU> yt(xt.dim, xt.shape);
	dtype* y_data = yt.buffer();
	dtype* x_data = xt.buffer();

	// reference implementation
	// reliable (but slow)

	const int length = xt.size();
	for (int k = 0; k < length; ++k)
	{
		dtype x = x_data[k];
		y_data[k] = x / ((dtype)1.0 + exp(-x));
	}

	return yt;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_silu(const Tensor<dtype, CUDA>& xt)
{
	// access the data arrays
	Tensor<dtype, CUDA> yt(xt.dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_silu_f32(xt, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}


/*
  GeLU activation (elementwise op).
  gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
  @param xt: input tensor
*/
template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> tensor_gelu(const Tensor<dtype, CPU>& xt, const bool approx=false)
{
	// access the data arrays
	Tensor<dtype, CPU> yt(xt.dim, xt.shape);
	dtype* y_data = yt.buffer();
	dtype* x_data = xt.buffer();

	// reference implementation
	// reliable (but slow)

	const int length = xt.size();

	if (!approx)
	{
		for (int k = 0; k < length; ++k)
		{
			dtype x = x_data[k];
			y_data[k] = x * (dtype)0.5 * ((dtype)1.0 + erf(x / sqrt((dtype)2.0)));
		}
	}
	else
	{
		for (int k = 0; k < length; ++k)
		{
			dtype x = x_data[k];
			y_data[k] = x * (dtype)0.5 * ((dtype)1.0 + tanh((dtype)sqrt(2.0 / 3.14159265358979323846) * (x + (dtype)0.044715 * x * x * x)));
		}
	}

	return yt;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gelu(const Tensor<dtype, CUDA>& xt, const bool approx = false)
{
	// access the data arrays
	Tensor<dtype, CUDA> yt(xt.dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_gelu_f32(xt, approx, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __MATH_OPS__
