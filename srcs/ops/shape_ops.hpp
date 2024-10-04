#ifndef __SHAPE_OPS__
#define __SHAPE_OPS__

#include "tensor.hpp"
#include "core_concepts.hpp"

/*
  View into the given buffer with a new tensor.
*/
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

/*
  Splitting over the first dimension into equal pieces.
*/
template<ArithmeticType dtype, Device device>
static std::vector<Tensor<dtype, device>> tensor_split(const Tensor<dtype, device>& x, const int32 splits)
{
	std::vector<Tensor<dtype, device>> ys(splits);
	for (size_t ix = 0; ix < ys.size(); ++ix)
	{
		auto& y = ys[ix];
		y = x;

		y.shape[0] /= splits;
		y.stride = calc_default_stride(y.dim, y.shape);
		y.name = "";

		y.offset += ix * y.stride[0] * y.shape[0] * sizeof(dtype);
	}
	return ys;
}

#endif  // __SHAPE_OPS__
