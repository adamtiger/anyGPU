#ifndef __TRANSP_OPS__
#define __TRANSP_OPS__

#include "transp_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


template<NotHalfFloatType dtype>
Tensor<dtype, CPU> tensor_transp(const Tensor<dtype, CPU>& x)
{
	ACASSERT(x.dim == 2, "transpose expects 2 dimensional tensors");

	int m = x.shape[0];
	int n = x.shape[1];

	dtype* x_data = x.buffer();

	std::vector<int> y_shape({ n, m });
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
Tensor<dtype, CUDA> tensor_transp(const Tensor<dtype, CUDA>& x)
{
	ACASSERT(x.dim == 2, "transpose expects 2 dimensional tensors");

	int m = x.shape[0];
	int n = x.shape[1];
	std::vector<int> y_shape({ n, m });
	Tensor<dtype, CUDA> y(y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		tensor_transp_f32(x, y);
	}
	else if constexpr (std::is_same_v<dtype, int8>)
	{
		tensor_transp_i8(x, y);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return y;
}

#endif  // __TRANSP_OPS__
