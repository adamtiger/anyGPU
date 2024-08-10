#include "tensor.hpp"

int calculate_default_size(const std::vector<int>& shape)
{
	int tensor_size = 1;
	for (int s : shape)
	{
		tensor_size *= s;
	}
	return tensor_size;
}

