#ifndef __MM_OPS__
#define __MM_OPS__

#include "mm_ops.cuh"
#include "shape_ops.hpp"

#include "tensor.hpp"
#include "core_concepts.hpp"


template<NotHalfFloatType dtype>
Tensor<dtype, CPU> tensor_mm(const Tensor<dtype, CPU>& lhs, const Tensor<dtype, CPU>& rhs)
{
	ACASSERT(matmul_compatible(lhs, rhs) == true, "matrix multiplication requires compatible matrices");

	int m = lhs.shape[0];
	int n = rhs.shape[1];
	int k = lhs.shape[1];

	dtype* lhs_data = lhs.buffer();
	dtype* rhs_data = rhs.buffer();

	std::vector<int> res_shape({m, n});
	Tensor<dtype, CPU> res(res_shape);
	dtype* res_data = res.buffer();

	// reference implementation
	// reliable (but slow)
	for (int r = 0; r < m; ++r)
	{
		for (int c = 0; c < n; ++c)
		{
			float32 acc = 0.f;
			int offs_out = r * res.stride[0] + c * res.stride[1];
			for (int l = 0; l < k; ++l)
			{
				int offs_lhs = r * lhs.stride[0] + l * lhs.stride[1];
				int offs_rhs = l * rhs.stride[0] + c * rhs.stride[1];
				
				acc += lhs_data[offs_lhs] * rhs_data[offs_rhs];
			}

			res_data[offs_out] = acc;
		}
	}

	return res;
}


template<typename dtype>
Tensor<dtype, CUDA> tensor_mm(const Tensor<dtype, CUDA>& lhs, const Tensor<dtype, CUDA>& rhs)
{
	ACASSERT(matmul_compatible(lhs, rhs) == true, "matrix multiplication requires compatible matrices");

	int m = lhs.shape[0];
	int n = rhs.shape[1];
	std::vector<int> res_shape({ m, n });
	Tensor<dtype, CUDA> res(res_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		opt1::tensor_mm_f32(lhs, rhs, res);
	}
	else if constexpr (std::is_same_v<dtype, float16>)
	{
		opt2::tensor_mm_f16(lhs, rhs, res);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return res;
}




template<NotHalfFloatType dtype>
Tensor<dtype, CPU> tensor_gemm(
	const Tensor<dtype, CPU>& xt,
	const Tensor<dtype, CPU>& wt,
	const Tensor<dtype, CPU>& bt)
{
	ACASSERT(matmul_compatible(xt, wt) == true, "matrix multiplication requires compatible matrices");
	ACASSERT(wt.shape[1] == bt.shape[0], "incorrect bias shape");

	int m = xt.shape[0];
	int n = wt.shape[1];
	int k = xt.shape[1];

	dtype* xt_data = xt.buffer();
	dtype* wt_data = wt.buffer();
	dtype* bt_data = bt.buffer();

	std::vector<int> res_shape({ m, n });
	Tensor<dtype, CPU> res(res_shape);
	dtype* res_data = res.buffer();

	// reference implementation
	// reliable (but slow)
	for (int r = 0; r < m; ++r)
	{
		for (int c = 0; c < n; ++c)
		{
			float32 acc = 0.f;
			int offs_out = r * res.stride[0] + c * res.stride[1];
			for (int l = 0; l < k; ++l)
			{
				int offs_lhs = r * xt.stride[0] + l * xt.stride[1];
				int offs_rhs = l * wt.stride[0] + c * wt.stride[1];

				acc += xt_data[offs_lhs] * wt_data[offs_rhs];
			}

			res_data[offs_out] = acc + bt_data[c];
		}
	}

	return res;
}


template<typename dtype>
Tensor<dtype, CUDA> tensor_gemm(
	const Tensor<dtype, CUDA>& xt, 
	const Tensor<dtype, CUDA>& wt,
	const Tensor<dtype, CUDA>& bt)
{
	ACASSERT(matmul_compatible(xt, wt) == true, "matrix multiplication requires compatible matrices");

	int m = xt.shape[0];
	int n = wt.shape[1];
	std::vector<int> res_shape({ m, n });
	Tensor<dtype, CUDA> res(res_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		tensor_gemm_f32_opt0(xt, wt, bt, res);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return res;
}




template<FloatingPointType dtype, Device device>
Tensor<dtype, device> tensor_linear(
	const Tensor<dtype, device>& xt,  // (*, in_features)
	const Tensor<dtype, device>& wt,  // (in_features, out_features)
	const Tensor<dtype, device>& bt)  // (out_features)
{
	// view xt as a 2d matrix
	int d = xt.dim;
	int cols = xt.shape[d - 1];
	int rows = xt.numel() / cols;
	Tensor<dtype, device> x_as_mtx = tensor_view(xt, { rows, cols });
	
	// linear transformation
	auto xw = tensor_gemm(x_as_mtx, wt, bt);

	// result should have the same head shape as the input
	// (head: ignoring the last dimension)
	std::vector<int> y_shape(d);
	for (int ix = 0; ix < d - 1; ++ix)
	{
		y_shape[ix] = xt.shape[ix];
	}
	y_shape[d - 1] = xw.shape[1];

	auto y = tensor_view(xw, y_shape);
	
	return y;
}

#endif  // __MM_OPS__
