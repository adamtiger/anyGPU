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
inline Tensor<trg_dtype, CPU> tensor_quantize_linear(const Tensor<src_dtype, CPU>& x, const src_dtype scale, const trg_dtype bias)
{
	// access the data arrays
	Tensor<trg_dtype, CPU> y(x.dim, x.shape);
	trg_dtype* y_data = y.buffer();
	src_dtype* x_data = x.buffer();

	// reference implementation
	// reliable (but slow)

	int32 bias_i32 = static_cast<int32>(bias);
	constexpr int32 lowest = static_cast<int32>(std::numeric_limits<trg_dtype>::lowest());
	constexpr int32 highest = static_cast<int32>(std::numeric_limits<trg_dtype>::max());

	int n = x.size();
	for (int ix = 0; ix < n; ++ix)
	{
		// transformation to integer
		int32 qx = static_cast<int32>(round(x_data[ix] / scale)) + bias_i32;

		// saturation
		qx = std::min(std::max(qx, lowest), highest);

		// saving result
		y_data[ix] = (int8)qx;
	}

	return y;
}


template<FloatingPointType src_dtype, IntegerType trg_dtype>
inline Tensor<trg_dtype, CUDA> tensor_quantize_linear(const Tensor<src_dtype, CUDA>& x, const src_dtype scale, const trg_dtype bias)
{
	Tensor<trg_dtype, CUDA> y(x.dim, x.shape);

	if constexpr (std::is_same_v<src_dtype, float32> && std::is_same_v<trg_dtype, int8>)
	{
		cu_tensor_quantize_linear_cuda_f32_i8(x, scale, bias, y);
	}
	else
	{
		static_assert(std::is_same_v<src_dtype, float32> && std::is_same_v<trg_dtype, int8>, "Unsupported data types");
	}

	return y;
}


/*
*  Linearly dequantizes the input tensor
*  from lower precision to higher precision.
*  This is a per-tensor dequantization.
*/
template<IntegerType src_dtype, PreciseFloatType trg_dtype>
inline Tensor<trg_dtype, CPU> tensor_dequantize_linear(const Tensor<src_dtype, CPU>& x, const trg_dtype scale, const src_dtype bias)
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


template<IntegerType src_dtype, FloatingPointType trg_dtype>
inline Tensor<trg_dtype, CUDA> tensor_dequantize_linear(const Tensor<src_dtype, CUDA>& x, const trg_dtype scale, const src_dtype bias)
{
	Tensor<trg_dtype, CUDA> y(x.dim, x.shape);

	if constexpr (std::is_same_v<src_dtype, int8> && std::is_same_v<trg_dtype, float32>)
	{
		cu_tensor_dequantize_linear_cuda_i8_f32(x, scale, bias, y);
	}
	else
	{
		static_assert(std::is_same_v<src_dtype, int8> && std::is_same_v<trg_dtype, float32>, "Unsupported data types");
	}

	return y;
}


/*
   Quantized matrix multiplication.
   Calculates the matmul on the low precision tensors,
   and calculates the quantized output.
*/
template<IntegerType lp_dtype, PreciseFloatType hp_dtype>
inline Tensor<lp_dtype, CPU> tensor_qmm(
	const Tensor<lp_dtype, CPU>& a,
	const Tensor<lp_dtype, CPU>& b,
	const hp_dtype sa, const lp_dtype zpa,
	const hp_dtype sb, const lp_dtype zpb, 
	const hp_dtype sy, const lp_dtype zpy)
{
	assert(matmul_compatible(a, b));

	int64 m = a.shape[0];
	int64 n = b.shape[1];
	int64 k = a.shape[1];

	// access the data arrays
	std::vector<int64> y_shape({ m, n });
	Tensor<lp_dtype, CPU> y(y_shape);
    lp_dtype* y_data = y.buffer();
	lp_dtype* a_data = a.buffer();
	lp_dtype* b_data = b.buffer();

	// reference implementation
	// reliable (but slow)
	constexpr int32 lowest = static_cast<int32>(std::numeric_limits<lp_dtype>::lowest());
	constexpr int32 highest = static_cast<int32>(std::numeric_limits<lp_dtype>::max());

	hp_dtype s = (sa * sb) / sy;
	hp_dtype zpy_hp = static_cast<hp_dtype>(zpy);
	int32 zpa_i32 = static_cast<int32>(zpa);
	int32 zpb_i32 = static_cast<int32>(zpb);

	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			int32 accumulator = 0;
			for (int c = 0; c < k; ++c)
			{
				accumulator += ((int32)(a_data[i * k + c]) - zpa) * ((int32)(b_data[c * n + k]) - zpb);
			}

			int32 qx = static_cast<int32>(round(static_cast<hp_dtype>(accumulator) * s + zpy_hp));
			qx = std::min(std::max(qx, lowest), highest);

			y_data[i * n + j] = qx;
		}
	}

	return y;
}


template<IntegerType lp_dtype, FloatingPointType hp_dtype>
inline Tensor<lp_dtype, CUDA> tensor_qmm(
	const Tensor<lp_dtype, CUDA>& a,
	const Tensor<lp_dtype, CUDA>& b,
	const hp_dtype sa, const lp_dtype zpa,
	const hp_dtype sb, const lp_dtype zpb,
	const hp_dtype sy, const lp_dtype zpy)
{
	assert(matmul_compatible(a, b));

	int64 m = a.shape[0];
	int64 n = b.shape[1];
	int64 k = a.shape[1];

	// access the data arrays
	std::vector<int64> y_shape({ m, n });
	Tensor<lp_dtype, CUDA> y(y_shape);

	if constexpr (std::is_same_v<lp_dtype, int8> && std::is_same_v<hp_dtype, float32>)
	{
		cu_tensor_qmm_cuda_i8_f32(a, b, sa, zpa, sb, zpb, sy, zpy, y);
	}
	else
	{
		static_assert(std::is_same_v<lp_dtype, int8> && std::is_same_v<hp_dtype, float32>, "Unsupported data types");
	}

	return y;
}

#endif  // __QUANTIZE_OPS__
