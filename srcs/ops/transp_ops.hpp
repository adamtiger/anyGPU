#ifndef __TRANSP_OPS__
#define __TRANSP_OPS__

#include "transp_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/* tranpose for 2d tensors */

template<NotHalfFloatType dtype>
inline Tensor<dtype, CPU> tensor_transp(const Tensor<dtype, CPU>& x)
{
	ACASSERT(x.dim == 2, "transpose expects 2 dimensional tensors");

	int m = x.shape[0];
	int n = x.shape[1];

	dtype* x_data = x.buffer();

	std::vector<int64> y_shape({ n, m });
	Tensor<dtype, CPU> y(y_shape);
	dtype* y_data = y.buffer();

	// reference implementation
	// reliable (but slow)

	int x_stride_r = x.stride[0];
	int x_stride_c = x.stride[1];

	int y_stride_r = y.stride[0];
	int y_stride_c = y.stride[1];

	for (int r = 0; r < m; ++r)
	{
		for (int c = 0; c < n; ++c)
		{
			int y_offset = c * y_stride_r + r * y_stride_c;
			int x_offset = r * x_stride_r + c * x_stride_c;

			y_data[y_offset] = x_data[x_offset];
		}
	}

	return y;
}


template<typename dtype>
inline Tensor<dtype, CUDA> tensor_transp(const Tensor<dtype, CUDA>& x)
{
	ACASSERT(x.dim == 2, "transpose expects 2 dimensional tensors");

	int m = x.shape[0];
	int n = x.shape[1];
	std::vector<int64> y_shape({ n, m });
	Tensor<dtype, CUDA> y(y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_transp_f32(x, y);
	}
	else if constexpr (std::is_same_v<dtype, int8>)
	{
		cu_tensor_transp_i8(x, y);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return y;
}


/* tranpose nd tensors, by swapping two axes */

template<NotHalfFloatType dtype>
inline Tensor<dtype, CPU> tensor_transp(const Tensor<dtype, CPU>& x, const int32 ax1, const int32 ax2)
{
	int32 dim = x.dim;
	Shape y_shape = x.shape;
	std::swap(y_shape[ax1], y_shape[ax2]);
	Tensor<dtype, CPU> y(dim, y_shape);

	dtype* x_data = x.buffer();
	dtype* y_data = y.buffer();

	// reference implementation
	// reliable (but slow)

	int num_elements = x.numel();
	Index x_index{};
	for (int ix = 0; ix < num_elements; ++ix)
	{
		int x_offset = calculate_offset(dim, x.stride, x_index);

		Index& y_index = x_index;
		std::swap(y_index[ax1], y_index[ax2]);
		int y_offset = calculate_offset(dim, y.stride, y_index);
		
		y_data[y_offset] = x_data[x_offset];

		std::swap(y_index[ax1], y_index[ax2]);
		increment_index(dim, x.shape, x_index);
	}

	return y;
}


template<typename dtype>
inline Tensor<dtype, CUDA> tensor_transp(const Tensor<dtype, CUDA>& xt, const int32 ax1, const int32 ax2)
{
	int64 dim = xt.dim;
	Shape y_shape = xt.shape;
	std::swap(y_shape[ax1], y_shape[ax2]);
	Tensor<dtype, CUDA> yt(dim, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		// assumes contigous memory
		cu_tensor_transp_swap_f32(xt, ax1, ax2, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return yt;
}

#endif  // __TRANSP_OPS__
