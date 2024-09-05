#ifndef __QUANTIZE_OPS__
#define __QUANTIZE_OPS__

#include "quantize_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"
#include <limits>

/*
*  Linearly quantizes the input tensor
*  from higher precision to lower precision.
*  This is a per-tensor quantization.
*/
template<PreciseFloatType src_dtype, IntegerType trg_dtype>
Tensor<trg_dtype, CPU> tensor_quantize_linear(const Tensor<src_dtype, CPU>& x, const src_dtype scale, const trg_dtype bias)
{
	// access the data arrays
	Tensor<trg_dtype, CPU> y(x.dim, x.shape);
	trg_dtype* y_data = y.buffer();
	src_dtype* x_data = x.buffer();

	// reference implementation
	// reliable (but slow)

	int32 bias_i32 = static_cast<int32>(bias);
	int32 lowest = static_cast<int32>(std::numeric_limits<trg_dtype>::lowest());
	int32 highest = static_cast<int32>(std::numeric_limits<trg_dtype>::max());

	int n = x.size();
	for (int ix = 0; ix < n; ++ix)
	{
		// transformation to integer
		int32 qx = static_cast<int32>(round(x_data[ix] / scale)) + bias_i32;

		// saturation
		qx = std::min(std::max(qx, lowest), highest);

		// saving result
		y_data[ix] = qx;
	}

	return y;
}


template<typename dtype>
Tensor<dtype, CUDA> tensor_quantize_linear(const Tensor<dtype, CUDA>& x)
{
	assert(x->dim == 2);

	int m = x.shape[0];
	int n = x.shape[1];
	std::vector<int> y_shape({ m, n });
	Tensor<dtype, CUDA> y(y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		tensor_softmax_f32(x, y);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return y;
}


/*
*  Linearly dequantizes the input tensor
*  from lower precision to higher precision.
*  This is a per-tensor dequantization.
*/
template<IntegerType src_dtype, PreciseFloatType trg_dtype>
Tensor<trg_dtype, CPU> tensor_dequantize_linear(const Tensor<src_dtype, CPU>& x, const trg_dtype scale, const src_dtype bias)
{
	// access the data arrays
	Tensor<trg_dtype, CPU> y(x.dim, x.shape);
	trg_dtype* y_data = y.buffer();
	src_dtype* x_data = x.buffer();

	// reference implementation
	// reliable (but slow)
	int32 bias_i32 = static_cast<int32>(bias);

	int n = x.size();
	for (int ix = 0; ix < n; ++ix)
	{
		// transformation to integer
		trg_dtype qx = static_cast<trg_dtype>(static_cast<int32>(x_data[ix]) - bias_i32) * scale;

		// saving result
		y_data[ix] = qx;
	}

	return y;
}

#endif  // __QUANTIZE_OPS__
