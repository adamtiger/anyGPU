#ifndef __MM_OPS__
#define __MM_OPS__

#include "mm_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


template<NotHalfFloatType dtype>
Tensor<dtype, CPU> tensor_mm(const Tensor<dtype, CPU>& lhs, const Tensor<dtype, CPU>& rhs)
{
	assert(matmul_compatible(lhs, rhs));

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
	assert(matmul_compatible(lhs, rhs));

	int m = lhs.shape[0];
	int n = rhs.shape[1];
	std::vector<int> res_shape({ m, n });
	Tensor<dtype, CUDA> res(res_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		tensor_mm_f32(lhs, rhs, res);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return res;
}

#endif  // __MM_OPS__
