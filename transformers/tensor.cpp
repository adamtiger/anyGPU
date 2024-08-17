#include "tensor.hpp"

int calc_default_size(const std::vector<int>& shape)
{
	int tensor_size = 1;
	for (int s : shape)
	{
		tensor_size *= s;
	}
	return tensor_size;
}

std::vector<int> calc_default_stride(const std::vector<int>& shape)
{
	int dim = calc_dim(shape);
	std::vector<int> stride(dim);
	stride[dim - 1] = 1;
	for (int dix = dim - 2; dix >= 0; --dix)
	{
		stride[dix] = stride[dix + 1] * shape[dix + 1];
	}
	return stride;
}

std::array<int, MAX_TENSOR_DIM> cvt_vector2array(const std::vector<int>& v)
{
	std::array<int, MAX_TENSOR_DIM> arr = {};
	int dim = calc_dim(v);
	for (int dix = 0; dix < dim; ++dix)
	{
		arr[dix] = v[dix];
	}
	return arr;
}

int calc_dim(const std::vector<int>& v)
{
	return static_cast<int>(v.size());
}

std::string represent_array(const int dim, const std::array<int, MAX_TENSOR_DIM>& arr)
{
	std::stringstream ss;

	ss << "[";
	for (int ix = 0; ix < dim-1; ++ix)
	{
		ss << arr[ix] << ", ";
	}
	ss << arr[dim - 1];
	ss << "]";

	return ss.str();
}
