#ifndef __BINARY_OPS__
#define __BINARY_OPS__

#include "tensor.hpp"
#include <cassert>

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


#endif  // __BINARY_OPS__
