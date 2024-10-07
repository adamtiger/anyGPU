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

/*
  Concatenating over the last dimension of the given vector.
  Assumes contigous memory.
*/
template<ArithmeticType dtype, Device device>
static Tensor<dtype, device> tensor_concat(const Tensor<dtype, device>& x1, const Tensor<dtype, device>& x2)
{
	ACASSERT(x1.dim == x2.dim, "dimensions have to be equal");

	int y_dim = x1.dim;

	ACASSERT(y_dim >= 2, "tensor needs to be at least 2 dim");

	for (int ix = 0; ix < y_dim - 1; ++ix)
	{
		ACASSERT(x1.shape[ix] == x2.shape[ix], "non concatenated axis should have the same length");
	}
	
	Shape y_shape = x1.shape;
	y_shape[y_dim - 1] = x1.shape[y_dim - 1] + x2.shape[y_dim - 1];
	Tensor<dtype, device> y(y_dim, y_shape);
	
	// device specific implementations

	dtype* x1_data = x1.buffer();
	dtype* x2_data = x2.buffer();
	dtype* y_data = y.buffer();

	int x1_lastd_stride = x1.stride[y_dim - 2];
	int x2_lastd_stride = x2.stride[y_dim - 2];
	int y_lastd_stride = y.stride[y_dim - 2];

	int x1_lastd_size = x1.shape[y_dim - 1];
	int x1_lastd_byte_size = x1_lastd_size * sizeof(dtype);
	int x2_lastd_byte_size = x2.shape[y_dim - 1] * sizeof(dtype);

	int num_slices = x1.size() / x1_lastd_stride;
	for (int nix = 0; nix < num_slices; ++nix)
	{
		if constexpr (device == CPU)
		{
			memcpy(
				y_data + nix * y_lastd_stride, 
				x1_data + nix * x1_lastd_stride, 
				x1_lastd_byte_size);

			memcpy(
				y_data + nix * y_lastd_stride + x1_lastd_size,
				x2_data + nix * x2_lastd_stride, 
				x2_lastd_byte_size);
		}
		else if constexpr (device == CUDA)
		{
			cudaMemcpy(
				y_data + nix * y_lastd_stride,
				x1_data + nix * x1_lastd_stride,
				x1_lastd_byte_size,
				cudaMemcpyDeviceToDevice);

			cudaMemcpy(
				y_data + nix * y_lastd_stride + x1_lastd_size,
				x2_data + nix * x2_lastd_stride,
				x2_lastd_byte_size,
				cudaMemcpyDeviceToDevice);
		}
		else
		{
			static_assert(device == CUDA, "unknown device");
		}
	}

	return y;
}

#endif  // __SHAPE_OPS__
