#ifndef __SHAPE_OPS__
#define __SHAPE_OPS__

#include "tensor.hpp"
#include "core_concepts.hpp"

template<ArithmeticType dtype, Device device>
static Tensor<dtype, device> tensor_view(const Tensor<dtype, device>& x, const std::vector<int>& new_shape)
{
	Tensor<dtype, device> y = x;
	y.id = GlobalUUIDGenerator::generate_id();
	y.dim = static_cast<int>(new_shape.size());
	y.shape = cvt_vector2array(new_shape);
	y.stride = cvt_vector2array(calc_default_stride(new_shape));
	y.name = "";
	return y;
}

#endif  // __SHAPE_OPS__
