#ifndef __BINARY_OPS__
#define __BINARY_OPS__

#include "binary_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


template<NotHalfFloatType dtype>
Tensor<dtype, CPU> tensor_add(const Tensor<dtype, CPU>& lhs, const Tensor<dtype, CPU>& rhs)
{
	assert(elementwise_compatible(lhs, rhs));

	int length = lhs.size();
	dtype* lhs_data = lhs.buffer();
	dtype* rhs_data = rhs.buffer();

	Tensor<dtype, CPU> res(lhs.dim, lhs.shape, lhs.stride);
	dtype* res_data = res.buffer();

	for (int ix = 0; ix < length; ++ix)
	{
		res_data[ix] = lhs_data[ix] + rhs_data[ix];
	}

	return res;
}


template<typename dtype>
Tensor<dtype, CUDA> tensor_add(const Tensor<dtype, CUDA>& lhs, const Tensor<dtype, CUDA>& rhs)
{
	assert((elementwise_compatible(lhs, rhs)==true));

	int length = lhs.size();
	auto kpms = calc_kernel_prms_pointwise(lhs);

	Tensor<dtype, CUDA> res(lhs.dim, lhs.shape, lhs.stride);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		tensor_add_f32(kpms, lhs, rhs, res);
	}
	else if constexpr (std::is_same_v<dtype, int32>)
	{
		tensor_add_i32(kpms, lhs, rhs, res);
	}
	else
	{
		std::cout << "Not implemented yet \n";
	}

	return res;
}

#endif  // __BINARY_OPS__
